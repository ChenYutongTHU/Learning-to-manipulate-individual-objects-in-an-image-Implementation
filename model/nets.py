import tensorflow as tf
from .utils.convolution_utils import gen_conv, gen_deconv, conv, deconv
from .utils.convolution_utils import _dilated_conv2d, conv2d, deconv2d, conv2d_transpose, InstanceNorm, fully_connect
from tensorflow.contrib.slim.nets import resnet_v2
from .utils.generic_utils import bin_edge_map, erode_dilate
from .utils.loss_utils import region_error
import math
import numpy as np

def Generator_forward(images, dataset, num_mask, model='resnet_v2_50', scope='Generator', reuse=None, training=True):
    if 'resnet' in model:
        generated_masks, logits = generator_resnet(images, dataset, num_mask,
            reuse=reuse, training=training, scope=scope, model=model)  #params 2e7
    else:
        generated_masks, logits = generator_segnet(images, num_mask, 
            scope=scope, reuse=reuse, training=training)
    return tf.expand_dims(generated_masks, axis=-2), tf.expand_dims(logits, axis=-2)  # B H W 1 C B H W 1 C

def Inpainter_forward(num_branch, input_masks, images, scope, dataset, reuse=None, training=True):
    edge_map = bin_edge_map(images, dataset)
    pred_intensities = []
    for m in range(num_branch):
        mask = input_masks[:,:,:,:,m]  #B H W 1
        reuse = True if reuse==True else (m>0)
        pred_intensity = inpaint_net(images, mask, edge_map, scope=scope, reuse=reuse, training=training)
        pred_intensities.append(pred_intensity)
    unconditioned_mean = inpaint_net(tf.zeros_like(images), tf.ones_like(mask), 
        edge_map, scope=scope, reuse=True, training=training)
    pred_intensities = tf.stack(pred_intensities, axis=-1) #B H W 3 C
    return pred_intensities, unconditioned_mean, edge_map

def inpaint_net(image, mask, edge, scope, reuse=None, training=True): #params 3e6<f=0.25>
    # intensity_masked
    # B H W 3
    image = image - 0.5 #0~1 -> -0.5~0.5
    intensity_masked = image*(1-mask)
    orisize = intensity_masked.get_shape().as_list()[1:-1] #[H,W]
    C = intensity_masked.get_shape().as_list()[-1] # 3
    f=0.5
    #edge B H W 2
    edge_in_channels = edge.get_shape().as_list()[-1] #2

    ones_x = tf.ones_like(intensity_masked)[:, :, :, 0:1] # B H W 1
    intensity_masked = tf.concat([intensity_masked, ones_x, 1-mask], axis=3) # B H W C+2
    intensity_in_channels = intensity_masked.get_shape().as_list()[-1]

    with tf.variable_scope(scope, reuse=reuse):

        aconv1 = conv( edge,    'aconv1', shape=[7,7, edge_in_channels,  int(64*f)],  stride=2, reuse=reuse, training=training ) # h/2(192), 64
        aconv2 = conv( aconv1,  'aconv2', shape=[5,5,int(64*f), int(128*f)],  stride=2, reuse=reuse, training=training ) # h/4(96),  128
        aconv3 = conv( aconv2,  'aconv3', shape=[5,5,int(128*f),int(256*f)],  stride=2, reuse=reuse, training=training ) # h/8(48),  256
        aconv31= conv( aconv3, 'aconv31', shape=[3,3,int(256*f),int(256*f)],  stride=1, reuse=reuse, training=training )
        aconv4 = conv( aconv31, 'aconv4', shape=[3,3,int(256*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/16(24), 512
        aconv41= conv( aconv4, 'aconv41', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
        aconv5 = conv( aconv41, 'aconv5', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/32(12), 512
        aconv51= conv( aconv5, 'aconv51', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
        aconv6 = conv( aconv51, 'aconv6', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/64(6),  512

        bconv1 = conv( intensity_masked,    'bconv1', shape=[7,7, intensity_in_channels,  int(64*f)],  stride=2, reuse=reuse, training=training ) # h/2(192), 64
        bconv2 = conv( bconv1,  'bconv2', shape=[5,5,int(64*f), int(128*f)],  stride=2, reuse=reuse, training=training ) # h/4(96),  128
        bconv3 = conv( bconv2,  'bconv3', shape=[5,5,int(128*f),int(256*f)],  stride=2, reuse=reuse, training=training ) # h/8(48),  256
        bconv31= conv( bconv3, 'bconv31', shape=[3,3,int(256*f),int(256*f)],  stride=1, reuse=reuse, training=training )
        bconv4 = conv( bconv31, 'bconv4', shape=[3,3,int(256*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/16(24), 512
        bconv41= conv( bconv4, 'bconv41', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
        bconv5 = conv( bconv41, 'bconv5', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/32(12), 512
        bconv51= conv( bconv5, 'bconv51', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
        bconv6 = conv( bconv51, 'bconv6', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/64(6),  512

        #conv6 = tf.add( aconv6, bconv6 )
        conv6 = tf.concat( (aconv6, bconv6), 3 )  #h/64(6) 512*2*f
        outsz = bconv51.get_shape()                              # h/32(12), 512*f
        deconv5 = deconv( conv6, size=[outsz[1],outsz[2]], name='deconv5', shape=[4,4,int(512*2*f),int(512*f)], reuse=reuse, training=training )
        concat5 = tf.concat( (deconv5,bconv51,aconv51), 3 )              # h/32(12), 512*3*f

        intensity5 = conv( concat5, 'intensity5', shape=[3,3,int(512*3*f),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/32(12), C
        outsz = bconv41.get_shape()                              # h/16(24), 512*f
        deconv4 = deconv( concat5, size=[outsz[1],outsz[2]], name='deconv4', shape=[4,4,int(512*3*f),int(512*f)], reuse=reuse, training=training )
        upintensity4 = deconv( intensity5,   size=[outsz[1],outsz[2]], name='upintensity4', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
        concat4 = tf.concat( (deconv4,bconv41,aconv41,upintensity4), 3 )      # h/16(24), 512*3*f+C

        intensity4 = conv( concat4, 'intensity4', shape=[3,3,int(512*3*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/16(24), C
        outsz = bconv31.get_shape()                              # h/8(48),  256*f
        deconv3 = deconv( concat4, size=[outsz[1],outsz[2]], name='deconv3', shape=[4,4,int(512*3*f+C),int(256*f)], reuse=reuse, training=training )
        upintensity3 = deconv( intensity4,   size=[outsz[1],outsz[2]], name='upintensity3', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
        concat3 = tf.concat( (deconv3,bconv31,aconv31,upintensity3), 3 )      # h/8(48),  256*3*f+C

        intensity3 = conv( concat3, 'intensity3', shape=[3,3,int(256*3*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/8(48), C
        outsz = bconv2.get_shape()                               # h/4(96),  128*f
        deconv2 = deconv( concat3, size=[outsz[1],outsz[2]], name='deconv2', shape=[4,4,int(256*3*f+C),int(128*f)], reuse=reuse, training=training )
        upintensity2 = deconv( intensity3,   size=[outsz[1],outsz[2]], name='upintensity2', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
        concat2 = tf.concat( (deconv2,bconv2,aconv2,upintensity2), 3 )       # h/4(96),  128*3*f+C

        intensity2 = conv( concat2, 'intensity2', shape=[3,3,int(128*3*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/4(96), C
        outsz = bconv1.get_shape()                               # h/2(192), 64*f
        deconv1 = deconv( concat2, size=[outsz[1],outsz[2]], name='deconv1', shape=[4,4,int(128*3*f+C),int(64*f)], reuse=reuse, training=training )
        upintensity1 = deconv( intensity2,   size=[outsz[1],outsz[2]], name='upintensity1', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
        concat1 = tf.concat( (deconv1,bconv1,aconv1,upintensity1), 3 )       # h/2(192), 64*3*f+C

        intensity1 = conv( concat1, 'intensity1', shape=[5,5,int(64*3*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/2(192), C
        pred_intensity = tf.image.resize_images(intensity1, size=orisize)

        pred_intensity = pred_intensity + 0.5
        return pred_intensity


def generator_resnet(images, dataset , num_mask, scope, model='resnet_v2_50', reuse=None, training=True):
    #images = 0~1
    images = (images-0.5)*2 #-1 ~ 1
    assert dataset in ['multi_texture','flying_animals','multi_dsprites','objects_room']
    if dataset in ['multi_texture', 'multi_dsprites','objects_room']:
        images = tf.image.resize_images(images, size=(128,128)) #64*64 -> 128*128
    #pad to 32k+1
    x = tf.pad(images, paddings=[[0,0],[0,1],[0,1],[0,0]], mode='REFLECT') 
    dilations = [6, 12, 18, 24] if dataset=='flying_animals' else [2,4,6,8]
    o = []
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if model=='resnet_v2_50': 
          net, end_points = resnet_v2.resnet_v2_50(x, None, is_training=training, global_pool=False, output_stride=4, reuse=reuse, scope=None)
        elif model=='resnet_v2_101':
          net, end_points = resnet_v2.resnet_v2_101(x, None, is_training=training, global_pool=False, output_stride=4, reuse=reuse, scope=None)
        else:
          raise IOError("Only resnet_v2_50 or resnet_v2_101 available")
    #classfication
    with tf.compat.v1.variable_scope(scope, reuse=tf.AUTO_REUSE):  
        for i, d in enumerate(dilations):
            o.append(_dilated_conv2d(net, 3, num_mask, d, name='aspp/conv%d' % (i+1), biased=True))
        logits = tf.add_n(o)

    if dataset == 'flying_animals':
        logits = tf.image.resize_images(logits, size=(193,257))  #B H W C align the feature
        logits = logits[:,:-1,:-1,:] #192 256 
    else:
        logits = tf.image.resize_images(logits, size=(129,129))  
        logits = logits[:,:-1,:-1,:] #128 128
        logits = tf.image.resize_images(logits, size=(64,64)) #64 64
    generated_masks = tf.nn.softmax(logits, axis=-1) # B H W cnum
    return generated_masks, logits

def generator_segnet(images, num_mask, scope, reuse=None, training=True, div=10.0):  #cnum=32 #params1.5e6
    """Mask network.
    Args:
        image: input rgb image [0, 1]
        num_mask: number of mask
    Returns:
        mask: mask region [0, 1], 1 is fully masked, 0 is not.  *num_mask
    """

    mask_channels = num_mask # probability of each mask
    x = images
    cnum = 64
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # stage1
        x_0 = gen_conv(x, cnum, 5, 1, name='conv1', training=training) # ---------------------------
        x   = gen_conv(x_0, 2*cnum, 3, 2, name='conv2_downsample', training=training) # Skip connection
        x_1 = gen_conv(x, 2*cnum, 3, 1, name='conv3', training=training) # -------------------
        x   = gen_conv(x_1, 4*cnum, 3, 2, name='conv4_downsample', training=training)
        x   = gen_conv(x, 4*cnum, 3, 1, name='conv5', training=training)
        x_2 = gen_conv(x, 4*cnum, 3, 1, name='conv6', training=training) # -----------------
        x   = gen_conv(x_2, 4*cnum, 3, rate=2, name='conv7_atrous', training=training)
        x   = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous', training=training)
        x   = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous', training=training)
        x   = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous', training=training)
        x   = gen_conv(x, 4*cnum, 3, 1, name='conv11', training=training) + x_2 #-------------
        x   = gen_conv(x, 4*cnum, 3, 1, name='conv12', training=training)
        x   = gen_deconv(x, 2*cnum, name='conv13_upsample', training=training)
        x   = gen_conv(x, 2*cnum, 3, 1, name='conv14', training=training) + x_1 # --------------------
        x   = gen_deconv(x, cnum, name='conv15_upsample', training=training) + x_0 #-------------------
        x   = gen_conv(x, cnum//2, 3, 1, name='conv16', training=training)
        x   = gen_conv(x, mask_channels, 3, 1, activation=tf.identity,
                     name='conv17', training=training)
        # Division by constant experimentally improved training
        x = tf.divide(x, tf.constant(div))   
        generated_mask = tf.nn.softmax(x, axis=-1)  #soft mask normalization  #B*H*W*num_mask
        return generated_mask, x 


def _sample_z(z_mean, z_log_sigma_sq):
    eps_shape = tf.shape(z_mean)
    eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
    # z = mu + sigma * epsilon
    z = tf.add(z_mean,tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z

def encoder(x, latent_dim, training=True):
    B, H, W, C = x.get_shape().as_list()
    conv1 = conv2d(x, filter_shape=[4,4,C,32], stride=2, padding='SAME', name='conv1', biased=True, dilation=1)
    conv1 = tf.nn.relu(conv1)

    conv2 = conv2d(conv1, filter_shape=[4,4,32,32], stride=2, padding='SAME', name='conv2', biased=True, dilation=1)
    conv2 = tf.nn.relu(conv2)

    conv3 = conv2d(conv2, filter_shape=[4,4,32,32], stride=2, padding='SAME', name='conv3', biased=True, dilation=1)
    conv3 = tf.nn.relu(conv3)

    conv4 = conv2d(conv3, filter_shape=[4,4,32,32], stride=2, padding='SAME', name='conv4', biased=True, dilation=1)
    conv4 = tf.nn.relu(conv4)
    #B 4 4 32

    flatten = tf.reshape(conv4, [-1,(H//16)*(W//16)*32])
    fc1 = fully_connect(flatten, weight_shape=[(H//16)*(W//16)*32, 256], name='fc1', biased=True)
    fc1 = tf.nn.relu(fc1)
    fc2 = fully_connect(fc1, weight_shape=[256, 256], name='fc2', biased=True)
    fc2 = tf.nn.relu(fc2)

    z_mean = fully_connect(fc2, weight_shape=[256, latent_dim], name='fc_zmean', biased=True, bias_init_value=0.0)
    z_log_sigma_sq = fully_connect(fc2, weight_shape=[256, latent_dim], name='fc_logsigmasq', biased=True, bias_init_value=0.0)
    return z_mean, z_log_sigma_sq

def decoder(z, output_ch, latent_dim, x, training=True):
    B, H, W, C = x.get_shape().as_list()
    up_fc2 = fully_connect(z, weight_shape=[latent_dim, 256], name='up_fc2', biased=True)
    up_fc2 = tf.nn.relu(up_fc2)

    up_fc1 = fully_connect(up_fc2, weight_shape=[256, (H//16)*(W//16)*32], name='up_fc1', biased=True)
    up_fc1 = tf.nn.relu(up_fc1)
    up_fc1 = tf.reshape(up_fc1, [-1,H//16,W//16,32])

    deconv4 = deconv2d(up_fc1, filter_shape=[4,4,32,32], output_size=[H//8,W//8], name='deconv4', padding='SAME', biased=True)
    deconv4 = tf.nn.leaky_relu(deconv4)

    deconv3 = deconv2d(deconv4, filter_shape=[4,4,32,32], output_size=[H//4,W//4], name='deconv3', padding='SAME', biased=True)
    deconv3 = tf.nn.leaky_relu(deconv3)

    deconv2 = deconv2d(deconv3, filter_shape=[4,4,32,32], output_size=[H//2,W//2], name='deconv2', padding='SAME', biased=True)
    deconv2 = tf.nn.leaky_relu(deconv2)    

    deconv1 = deconv2d(deconv2, filter_shape=[4,4,32,output_ch], output_size=[H,W], name='deconv1', padding='SAME', biased=True)
    
    out_logit =  tf.identity(deconv1)  
    return out_logit

def tex_mask_fusion(tex, mask): 
    inputs = tf.concat([tex, mask], axis=-1)#B H W (3+1)

    inputs = tf.pad(inputs, paddings=[[0,0],[1,2],[1,2],[0,0]], mode='REFLECT')
    conv1 = conv2d(inputs, filter_shape=[4,4,4,32], stride=1, padding='VALID', name='conv1', biased=True, dilation=1)
    conv1 = tf.nn.relu(conv1)

    conv1 = tf.pad(conv1, paddings=[[0,0],[1,2],[1,2],[0,0]], mode='REFLECT')
    conv2 = conv2d(conv1, filter_shape=[4,4,32,32], stride=1, padding='VALID', name='conv2', biased=True, dilation=1)
    conv2 = tf.nn.relu(conv2)    

    conv2 = tf.pad(conv2, paddings=[[0,0],[1,2],[1,2],[0,0]], mode='REFLECT')
    conv3 = conv2d(conv2, filter_shape=[4,4,32,3], stride=1, padding='VALID', name='conv3', biased=True, dilation=1)
    output = tf.nn.sigmoid(conv3)   

    return output, conv3

def encoder_decoder(x, output_ch, latent_dim,training=True):
    z_mean, z_log_sigma_sq = encoder(x, latent_dim, training=training)
    z = _sample_z(z_mean, z_log_sigma_sq)
    out_logit =  decoder(z, output_ch, latent_dim, x, training=training)
    return z_mean, z_log_sigma_sq, out_logit

def gaussian_kl(mean, log_sigma_sq):
    latent_loss = -0.5 * (1 + log_sigma_sq
                                   - tf.square(mean)
                                   - tf.exp(log_sigma_sq))  #B*Z_dim
    return latent_loss # B*z_dim

def VAE_forward(image, masks, bg_dim, tex_dim, mask_dim, scope='VAE', reuse=None, training=True, augmentation=False):
    B, H, W, C = image.get_shape().as_list()
    num_branch = masks.get_shape().as_list()[-1]  #B H W 1 M 

    with tf.compat.v1.variable_scope(scope, reuse=reuse):   
        tex_kl, out_texes = [],[]
        mask_kl, out_masks_logit = [], []
        fusion_error, out_fusion = [], []
        latent_zs = {'tex':[], 'mask':[], 'bg':None}

        for i in range(num_branch):
            

            inputs = image*masks[:,:,:,:,i] #B H W 3 
            with tf.compat.v1.variable_scope('separate/texVAE', reuse=tf.compat.v1.AUTO_REUSE):
                z_mean, z_log_sigma_sq, out_logit = encoder_decoder(inputs, output_ch=3, latent_dim=tex_dim, training=training)
            out_texes.append(tf.nn.sigmoid(out_logit)) #B,
            tex_kl.append(tf.reduce_mean(gaussian_kl(z_mean, z_log_sigma_sq), 0))  #B,dim -> dim
            latent_zs['tex'].append(z_mean)

            inputs = masks[:,:,:,:,i]
            with tf.compat.v1.variable_scope('separate/maskVAE', reuse=tf.compat.v1.AUTO_REUSE):
                z_mean, z_log_sigma_sq, out_logit = encoder_decoder(inputs, output_ch=1, latent_dim=mask_dim,  training=training) 
            out_masks_logit.append(out_logit)
            mask_kl.append(tf.reduce_mean(gaussian_kl(z_mean, z_log_sigma_sq), 0))           
            latent_zs['mask'].append(z_mean)

            #fuse tex and mask
            tex, mask = out_texes[-1], tf.nn.sigmoid(out_masks_logit[-1]) #B H W 3 B H W 1
            with tf.compat.v1.variable_scope('fusion', reuse=tf.compat.v1.AUTO_REUSE):
                fus_output, fus_output_logits = tex_mask_fusion(tex, mask) #B H W 3
            out_fusion.append(fus_output)
            error = tf.nn.sigmoid_cross_entropy_with_logits(labels=image*masks[:,:,:,:,i], logits=fus_output_logits) # B H W 3
            error = tf.reduce_mean(tf.reduce_sum(error, axis=[1,2,3]), axis=0) #B H W 3 -> B -> mean scalar
            fusion_error.append(error)

        #KL divergence loss
        tex_kl = tf.reduce_mean(tf.stack(tex_kl, axis=0), axis=0)# branch,dim -> dim, 

        #reconstruction error
        out_texes = tf.stack(out_texes, axis=-1) # B H W 3 M
        tex_error = tf.reduce_mean(region_error(X=out_texes, Y=image, region=masks)) #BHW3M BHW3 BHW1M ->B,M -> scalar

        out_masks_logit = tf.stack(out_masks_logit, axis=-1) #B H W 1 M
        out_masks = tf.nn.sigmoid(out_masks_logit) #B H W 1 M

        if not augmentation:
            mask_error_pixel = tf.nn.sigmoid_cross_entropy_with_logits(labels=masks, logits=out_masks_logit) #B H W 1 M
            mask_error_sum = tf.reduce_sum(mask_error_pixel, axis=[1,2,3]) #B,M
            mask_error = tf.reduce_mean(mask_error_sum)
            mask_kl = tf.reduce_mean(tf.stack(mask_kl, axis=0), axis=0)
        else:
            #-----------------data augmentation--------------
            #----------generate more position variation to help VAE decompose position feature---------------
            rep = 2
            aug_masks = tf.tile(masks, [2*rep,1,1,1,1]) #B H W 1 M -> 2*rep*B H W 1 M
            aug_masks = tf.transpose(aug_masks, perm=[0,4,1,2,3]) #2*rep*B M H W 1
            aug_masks = tf.reshape(aug_masks, [2*B*rep*num_branch,H,W,1])
            dx = tf.random.uniform(shape=[2*rep*B*num_branch,1],dtype=tf.dtypes.float32,minval=-1*30,maxval=30)
            dy = tf.random.uniform(shape=[2*rep*B*num_branch,1],dtype=tf.dtypes.float32,minval=-1*30,maxval=30)
            aug_masks = tf.contrib.image.translate(aug_masks,translations=tf.concat([dx,dy], axis=-1),interpolation='NEAREST')
            aug_masks = tf.random.shuffle(aug_masks, seed=None)
            inputs = aug_masks[0:rep*B*num_branch] #rep*B*M,H,W,1
            
            with tf.compat.v1.variable_scope('separate/maskVAE', reuse=tf.compat.v1.AUTO_REUSE):
                z_mean, z_log_sigma_sq, out_logit = encoder_decoder(inputs, output_ch=1, latent_dim=mask_dim,  training=training) 
            mask_error_pixel= tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=out_logit) #B*rep*M H W 1
            mask_error_sum = tf.reduce_sum(mask_error_pixel, axis=[1,2,3]) #B,M
            mask_error = tf.reduce_mean(mask_error_sum)           
            mask_kl= tf.reduce_mean(gaussian_kl(z_mean, z_log_sigma_sq), 0) #reduce batch         

        
        out_fusion = tf.stack(out_fusion, axis=-1) #B H W 3 M
        fusion_error = tf.reduce_mean(tf.stack(fusion_error, axis=0))  #average on all branch 

        # for background, we only learn the representation of its texture
        bg_mask = 1-tf.reduce_sum(masks, axis=-1)
        inputs = image*bg_mask #B H W 1
        with tf.compat.v1.variable_scope('separate/bgVAE', reuse=tf.compat.v1.AUTO_REUSE):
            z_mean, z_log_sigma_sq, out_logit = encoder_decoder(inputs, output_ch=3, latent_dim=bg_dim, training=training)
            out_bg = tf.nn.sigmoid(out_logit)
            bg_error = tf.reduce_mean(region_error(X=out_bg, Y=image, region=tf.expand_dims(bg_mask, axis=-1)))
            bg_kl= tf.reduce_mean(gaussian_kl(z_mean, z_log_sigma_sq), 0) #B,dim -> dim
            latent_zs['bg'] = z_mean

    loss = {'mask_kl': mask_kl, 'tex_kl': tex_kl, 'bg_kl':bg_kl, 
        'mask_error':mask_error, 'tex_error': tex_error, 'bg_error': bg_error , 'fusion_error': fusion_error}
    outputs = {'out_masks': out_masks, 'out_texes': out_texes, 'out_bg': out_bg, 'out_fusion': out_fusion}
    latent_zs['tex'] = tf.stack(latent_zs['tex'], axis=0) #branch-1, B, dim
    latent_zs['mask'] = tf.stack(latent_zs['mask'], axis=0) #branch-1 B dim

    return loss, outputs, latent_zs


def Fusion_forward(inputs, scope='Fusion', training=True, reuse=None):
    #inputs B H W N*C
    B, H, W, C = inputs.get_shape().as_list()
    with tf.compat.v1.variable_scope(scope, reuse=reuse):

        x = tf.pad(inputs, paddings=[[0,0],[1,2],[1,2],[0,0]], mode='REFLECT')
        conv1 = conv2d(x, filter_shape=[4,4,C,32], stride=1, padding='VALID', name='conv1', biased=True, dilation=1)
        conv1 = tf.nn.relu(conv1)

        conv1 = tf.pad(conv1, paddings=[[0,0],[1,2],[1,2],[0,0]], mode='REFLECT')
        conv2 = conv2d(conv1, filter_shape=[4,4,32,32], stride=1, padding='VALID', name='conv2', biased=True, dilation=1)
        conv2 = tf.nn.relu(conv2)    

        conv2 = tf.pad(conv2, paddings=[[0,0],[1,2],[1,2],[0,0]], mode='REFLECT')
        conv3 = conv2d(conv2, filter_shape=[4,4,32,32], stride=1, padding='VALID', name='conv3', biased=True, dilation=1)
        conv3 = tf.nn.relu(conv3)  


        conv3 = tf.pad(conv3, paddings=[[0,0],[1,2],[1,2],[0,0]], mode='REFLECT')
        conv4 = conv2d(conv3, filter_shape=[4,4,32,3], stride=1, padding='VALID', name='conv4', biased=True, dilation=1)
        out = tf.nn.sigmoid(conv4)  

        return out


def Perturbation_forward(var_num, image, generated_masks,VAE_outputs0, latent_zs, mask_top_dims, scope='VAE/'):
    B, H, W, C, M = generated_masks.get_shape().as_list()
    assert M==3
    mask_dim = latent_zs['mask'][0].get_shape().as_list()[1]
    tex_dim = latent_zs['tex'][0].get_shape().as_list()[1]
    k = tf.random.uniform(shape=[], dtype=tf.int32, minval=0, maxval=2)
    b = tf.cond(tf.math.equal(k,0), lambda:1, lambda:0)

    new_outs, new_labels = [], []
    for i in range(var_num):

        dim_x, dim_y = mask_top_dims[0], mask_top_dims[1]
        new_x = tf.random.uniform(shape=[B,1], dtype=tf.float32, minval=-1, maxval=1) #B,1
        new_y = tf.random.uniform(shape=[B,1], dtype=tf.float32, minval=-1, maxval=1) #B,1
        mask_z = tf.concat([latent_zs['mask'][k,:,:dim_x],
            new_x,
            latent_zs['mask'][k,:,dim_x+1:dim_y],
            new_y,
            latent_zs['mask'][k,:,dim_y+1:]], axis=-1)

        with tf.compat.v1.variable_scope(scope+'/separate/maskVAE', reuse=tf.compat.v1.AUTO_REUSE):
            new_mask_logit = decoder(z=mask_z, output_ch=1, latent_dim=mask_dim, x=image, training=True)
            new_mask = tf.nn.sigmoid(new_mask_logit) #B H W 1
        
        another_mask = VAE_outputs0['out_masks'][:,:,:,:,b]
        new_mask = (1-another_mask)*new_mask
        bg_mask = 1-new_mask-another_mask 
        new_masks = tf.cond(tf.math.equal(k,0), 
                            lambda:tf.stack([new_mask,another_mask,bg_mask], axis=-1), 
                            lambda:tf.stack([another_mask,new_mask,bg_mask], axis=-1))

        with tf.compat.v1.variable_scope(scope+'/fusion', reuse=tf.compat.v1.AUTO_REUSE):
            new_fus_output, null = tex_mask_fusion(VAE_outputs0['out_texes'][:,:,:,:,k], new_mask) #B H W 3 

        foregrounds = tf.stack([new_fus_output, VAE_outputs0['out_fusion'][:,:,:,:,b]], axis=-1) #B H W 3 M
        backgrounds = tf.expand_dims(VAE_outputs0['out_bg'], axis=-1)* \
                    tf.expand_dims(bg_mask, axis=-1)

        fusion_inputs = tf.concat([foregrounds, backgrounds], axis=-1) #B H W 3 fg_branch+1
        fusion_inputs = tf.reshape(fusion_inputs, [B,fusion_inputs.get_shape()[1],fusion_inputs.get_shape()[2],-1])
        fusion_outputs = Fusion_forward(inputs=fusion_inputs, scope='Fusion/', training=True, reuse=tf.compat.v1.AUTO_REUSE) #B H W 3
        new_outs.append(fusion_outputs)
        new_labels.append(new_masks)

    return new_outs, new_labels

