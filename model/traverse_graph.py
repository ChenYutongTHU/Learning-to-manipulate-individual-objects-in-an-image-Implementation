import tensorflow as tf  
import os
from .utils.generic_utils import train_op,myprint, myinput, tf_resize_imgs, reorder_mask, erode_dilate
from .nets import Generator_forward, encoder, decoder, tex_mask_fusion, Fusion_forward, gaussian_kl
import numpy as np

class Traverse_Graph(object):

    def __init__(self, FLAGS):
        self.config = FLAGS
        #load data
        self.batch_size = FLAGS.batch_size
        self.num_branch = FLAGS.num_branch
        self.img_height, self.img_width = FLAGS.img_height, FLAGS.img_width

        assert self.config.traverse_type in ['tex', 'mask', 'bg']
        self.traverse_type = self.config.traverse_type
        if self.traverse_type == 'bg':
            self.traverse_branch = [self.num_branch-1]
        else:
            self.traverse_branch = [i for i in range(0,self.num_branch) if self.config.traverse_branch=='all' or str(i) in self.config.traverse_branch.split(',')]

        ndim_dict = {'tex':self.config.tex_dim, 'mask':self.config.mask_dim, 'bg':self.config.bg_dim}
        self.ndim = ndim_dict[self.config.traverse_type]

        self.traverse_value = list(np.linspace(self.config.traverse_start, self.config.traverse_end, 60))
        self.VAE_outputs = []
        self.traverse_results =  []

    def build(self):        
        self.image_batch0 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,self.img_height,self.img_width,3])
        with tf.name_scope("Generator") as scope:
            self.generated_masks, null = Generator_forward(self.image_batch0, self.config.dataset, 
                    self.num_branch, model=self.config.model, training=False, reuse=None, scope=scope)

        filter_masks = []
        for i in range(self.config.num_branch):
            filter_masks.append(erode_dilate(self.generated_masks[:,:,:,:,i]))
        self.generated_masks = tf.stack(filter_masks, axis=-1)#B H W 1 num_branch
        self.generated_masks = reorder_mask(self.generated_masks)

        if self.config.dataset == 'flying_animals':
            #resize image to (96, 128)
            self.image_batch = tf_resize_imgs(self.image_batch0, size=[self.img_height//2,self.img_width//2])
            self.generated_masks = tf_resize_imgs(self.generated_masks, size=[self.img_height//2, self.img_width//2])
        else:
            self.image_batch = tf.identity(self.image_batch0)
            
        segmented_img = self.generated_masks*tf.expand_dims(self.image_batch, axis=-1)#B H W 3 M



        #Fusion
        bg_mask = 1-tf.reduce_sum(self.generated_masks[:,:,:,:,:-1*self.config.n_bg], axis=-1)
        with tf.compat.v1.variable_scope('VAE//separate/bgVAE', reuse=tf.compat.v1.AUTO_REUSE):
            bg_z_mean, bg_z_log_sigma_sq = encoder(x=bg_mask*self.image_batch, latent_dim=self.config.bg_dim, training=False)
            out_bg_logit = decoder(bg_z_mean, output_ch=3, latent_dim=self.config.bg_dim, x=bg_mask*self.image_batch, training=False)
            out_bg = tf.nn.sigmoid(out_bg_logit)

        self.out_bg = out_bg
        self.in_bg = bg_mask*self.image_batch
        for i in self.traverse_branch:
            with tf.compat.v1.variable_scope('VAE//separate', reuse=tf.compat.v1.AUTO_REUSE):
                with tf.compat.v1.variable_scope('texVAE', reuse=tf.compat.v1.AUTO_REUSE):
                    tex_z_mean, tex_z_log_sigma_sq = encoder(x=self.generated_masks[:,:,:,:,i]*self.image_batch, latent_dim=self.config.tex_dim, training=False)
                    out_tex_logit = decoder(tex_z_mean, output_ch=3, latent_dim=self.config.tex_dim, x=self.generated_masks[:,:,:,:,i]*self.image_batch, training=False)
                    out_tex = tf.nn.sigmoid(out_tex_logit)

                with tf.compat.v1.variable_scope('maskVAE', reuse=tf.compat.v1.AUTO_REUSE):
                    mask_z_mean, mask_z_log_sigma_sq = encoder(x=self.generated_masks[:,:,:,:,i], latent_dim=self.config.mask_dim, training=False)
                    out_mask_logit = decoder(mask_z_mean, output_ch=1, latent_dim=self.config.mask_dim, x=self.generated_masks[:,:,:,:,i], training=False)
                    out_mask = tf.nn.sigmoid(out_mask_logit)
            if self.traverse_type=='bg':
                output_ch = 3
                scope = 'VAE//separate/bgVAE'
                z_mean, z_log_sigma_sq = bg_z_mean, bg_z_log_sigma_sq
            elif self.traverse_type=='tex':
                output_ch = 3
                scope = 'VAE//separate/texVAE'
                z_mean, z_log_sigma_sq = tex_z_mean, tex_z_log_sigma_sq
            else:
                output_ch = 1
                scope = 'VAE//separate/maskVAE'
                z_mean, z_log_sigma_sq = mask_z_mean, mask_z_log_sigma_sq

            KLs = gaussian_kl(z_mean, z_log_sigma_sq)[0] #B,zdim -> zdim,
            self.traverse_dim = tf.math.top_k(KLs, k=self.config.top_kdim, sorted=True).indices

            for d_count in range(self.config.top_kdim):
                d = self.traverse_dim[d_count]
                delta_unit = tf.one_hot(indices=[d], depth=self.ndim)
                delta_unit = tf.tile(delta_unit, [self.batch_size,1]) #B,dim

                for k in self.traverse_value: 
                    shifted_z = z_mean + delta_unit*k #B,dim
                    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
                        out_logit = decoder(shifted_z, output_ch=output_ch, latent_dim=self.ndim, x=self.generated_masks[:,:,:,:,i], training=False)
                        out = tf.nn.sigmoid(out_logit)

                    if self.traverse_type=='bg':
                        foregrounds = segmented_img[:,:,:,:,:-self.config.n_bg] #B H W 3 C
                        backgrounds = tf.expand_dims(bg_mask*out, axis=-1) #B H W 3 1
                    else:
                        if self.traverse_type=='tex':
                            tex,mask = out, self.generated_masks[:,:,:,:,i]
                        else: #mask
                            tex,mask = out_tex, out 
                        with tf.compat.v1.variable_scope('VAE//fusion', reuse=tf.compat.v1.AUTO_REUSE):
                            VAE_fusion_out, null = tex_mask_fusion(tex=tex, mask=mask) 
                        foregrounds = tf.concat([segmented_img[:,:,:,:,0:i],
                            tf.expand_dims(VAE_fusion_out, axis=-1),
                            segmented_img[:,:,:,:,i+1:-self.config.n_bg]], axis=-1)
                        background_mask = 1-tf.reduce_sum(self.generated_masks[:,:,:,:,0:i],axis=-1)-mask-tf.reduce_sum(self.generated_masks[:,:,:,:,i+1:-self.config.n_bg],axis=-1)
                        backgrounds = tf.expand_dims(background_mask*out_bg, axis=-1)

                    fusion_inputs = tf.concat([foregrounds, backgrounds], axis=-1)
                    fusion_inputs = tf.reshape(fusion_inputs, [-1,fusion_inputs.get_shape()[1],fusion_inputs.get_shape()[2],3*fusion_inputs.get_shape()[4]])
                    fusion_output = Fusion_forward(inputs=fusion_inputs, scope='Fusion/', training=False, reuse=tf.compat.v1.AUTO_REUSE)
                    self.traverse_results.append(fusion_output)
    # def load_val_data(self):
    #     #now only support multi_dsprites
    #     if self.config.dataset == 'multi_texture':
    #         return multi_texture_utils.dataset(self.config.root_dir, val=True,
    #             batch_size=self.config.max_num, max_num=self.config.max_num, 
    #             zoom=('z' in self.config.variant), 
    #             rotation=('r' in self.config.variant), 
    #             texture_transform=self.config.texture_transform)
    #     elif self.config.dataset == 'flying_animals':
    #         return flying_animals_utils.dataset(self.config.root_dir,val=True,
    #             batch_size=self.config.max_num, max_num=self.config.max_num)
    #     elif self.config.dataset == 'multi_dsprites':
    #         return multi_dsprites_utils.dataset(self.config.root_dir,val=True,
    #             batch_size=self.batch_size, skipnum=0, takenum=self.config.skipnum,
    #             shuffle=False, map_parallel_calls=1)     
    #     elif self.config.dataset == 'objects_room':
    #         return objects_room_utils.dataset(self.config.root_dir,val=True,
    #             batch_size=self.batch_size, skipnum=0, takenum=self.config.skipnum,
    #             shuffle=False, map_parallel_calls=1)       
    #     else:
    #         raise IOError("Unknown Dataset")