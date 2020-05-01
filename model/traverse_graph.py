import tensorflow as tf  
import os
from data import multi_texture_utils, flying_animals_utils, multi_dsprites_utils, objects_room_utils
from .utils.generic_utils import train_op,myprint, myinput
from .nets import Generator_forward, encoder, decoder, tex_mask_fusion, Fusion_forward
import numpy as np

class Traverse_Graph(object):

    def __init__(self, FLAGS):
        self.config = FLAGS
        #load data
        self.batch_size = FLAGS.batch_size
        self.num_branch = FLAGS.num_branch
        self.img_height, self.img_width = FLAGS.img_height, FLAGS.img_width

        self.traverse_branch = [i for i in range(0,self.num_branch) if self.config.traverse_branch=='all' or str(i) in self.config.traverse_branch.split(',')]

        assert self.config.traverse_type in ['tex', 'mask']
        if self.num_branch-1 in self.traverse_branch:
            assert self.config.traverse_type == 'tex'

        self.traverse_type = self.config.traverse_type

        self.ndim = self.config.tex_dim if self.traverse_type=='tex' else self.config.mask_dim
        self.traverse_dim = [i for i in range(self.ndim) if self.config.traverse_dim=='all' or str(i) in self.config.traverse_dim.split(',')]

        self.traverse_value = list(range(-1*self.config.traverse_range, self.config.traverse_range+1))
        self.VAE_outputs = []
        self.traverse_results =  []

    def build(self):
        # dataset = self.load_val_data()
        # iterator = dataset.make_one_shot_iterator()
        # data_batch = iterator.get_next()
        # self.image_batch, self.GT_masks = data_batch['img'], data_batch['masks']

        # self.image_batch.set_shape([None, self.img_height, self.img_width, 3])
        
        self.image_batch = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,self.img_height,self.img_width,3])
        with tf.name_scope("Generator") as scope:
            self.generated_masks, null = Generator_forward(self.image_batch, self.config.dataset, 
                    self.num_branch, model=self.config.model, training=False, reuse=None, scope=scope)

        reordered = [self.generated_masks[:,:,:,:,i] for i in range(self.num_branch) if not i in self.config.bg] + \
                    [self.generated_masks[:,:,:,:,i] for i in self.config.bg]
        reordered = tf.stack(reordered, axis=-1) #B H W 1 M
        self.generated_masks = reordered
        #now only support multi_dsprites  (single channel for background)


        segmented_img = self.generated_masks*tf.expand_dims(self.image_batch, axis=-1)#B H W 3 M

        if self.traverse_type == 'tex':
            for i in self.traverse_branch:
                x = self.generated_masks[:,:,:,:,i]*self.image_batch
                scope_name = 'VAE//separate/texVAE_BG' if i >= self.num_branch-len(self.config.bg) else 'VAE//separate/texVAE'
                with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
                    z_mean, z_log_sigma_sq = encoder(x=x, latent_dim=self.config.bg_dim, training=False)
                    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq)) #B,dim

                for d in self.traverse_dim:
                    delta_unit = np.zeros([self.batch_size, self.config.tex_dim])
                    delta_unit[:,d] = 1
                    delta_unit = tf.constant(delta_unit, dtype=tf.float32) #B,dim

                    for k in self.traverse_value: 
                        shifted_z = z_mean + delta_unit*k*z_sigma*5 #B,dim
                        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
                            out_logit = decoder(shifted_z, output_ch=3, latent_dim=self.config.tex_dim, x=x, training=False)
                            out = tf.nn.sigmoid(out_logit)

                            new_img = self.image_batch*(1-self.generated_masks[:,:,:,:,i])+out*self.generated_masks[:,:,:,:,i]
                            self.traverse_results.append(new_img)
        elif self.traverse_type == 'mask':
            # background tex
            bg_mask = 1-tf.reduce_sum(self.generated_masks[:,:,:,:,:-1*len(self.config.bg)], axis=-1)
            with tf.compat.v1.variable_scope('VAE//separate/texVAE_BG', reuse=tf.compat.v1.AUTO_REUSE):
                bg_z_mean, null = encoder(x=bg_mask*self.image_batch, latent_dim=self.config.bg_dim, training=False)
                out_bg_logit = decoder(bg_z_mean, output_ch=3, latent_dim=self.config.bg_dim, x=bg_mask*self.image_batch, training=False)
                out_bg = tf.nn.sigmoid(out_bg_logit)

            for i in self.traverse_branch:
                with tf.compat.v1.variable_scope('VAE//separate', reuse=tf.compat.v1.AUTO_REUSE):
                    with tf.compat.v1.variable_scope('texVAE', reuse=tf.compat.v1.AUTO_REUSE):
                        tex_z_mean, tex_z_log_sigma_sq = encoder(x=self.generated_masks[:,:,:,:,i]*self.image_batch, latent_dim=self.config.tex_dim, training=False)
                        out_tex_logit = decoder(tex_z_mean, output_ch=3, latent_dim=self.config.tex_dim, x=self.generated_masks[:,:,:,:,i]*self.image_batch, training=False)
                        out_tex = tf.nn.sigmoid(out_tex_logit)

                    with tf.compat.v1.variable_scope('maskVAE', reuse=tf.compat.v1.AUTO_REUSE):
                        z_mean, z_log_sigma_sq = encoder(x=self.generated_masks[:,:,:,:,i], latent_dim=self.config.mask_dim, training=False)
                        z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq)) #B,dim               

                #traverse z_mean +- k*z_sigma{i==dim}
                for d in self.traverse_dim:
                    delta_unit = np.zeros([self.batch_size, self.config.mask_dim])
                    delta_unit[:,d] = 1
                    delta_unit = tf.constant(delta_unit, dtype=tf.float32) #B,dim

                    for k in self.traverse_value: 
                        shifted_z = z_mean + delta_unit*k*z_sigma*5 #B,dim
                        with tf.compat.v1.variable_scope('VAE//separate/maskVAE', reuse=tf.compat.v1.AUTO_REUSE):
                            out_logit = decoder(shifted_z, output_ch=1, latent_dim=self.config.mask_dim, x=self.generated_masks[:,:,:,:,i], training=False)
                            out_mask = tf.nn.sigmoid(out_logit)

                        with tf.compat.v1.variable_scope('VAE//fusion', reuse=tf.compat.v1.AUTO_REUSE):
                            VAE_fusion_out, null = tex_mask_fusion(tex=out_tex, mask=out_mask)

                        self.VAE_outputs.append(VAE_fusion_out)
                    
                        #self.traverse_results.append(out)  output no fusion


                        # approach 1: Other branches use the original segmented part
                        foregrounds = tf.concat([segmented_img[:,:,:,:,0:i],
                                 tf.expand_dims(VAE_fusion_out, axis=-1),
                                segmented_img[:,:,:,:,i+1:-1]], axis=-1) # B H W 3 num_branch-1
                        background_mask = 1-tf.reduce_sum(self.generated_masks[:,:,:,:,0:i],axis=-1)-out_mask-tf.reduce_sum(self.generated_masks[:,:,:,:,i+1:-1],axis=-1) #B H W 1
                        backgrounds = tf.expand_dims(background_mask*out_bg, axis=-1) #B H W 3 1
                        fusion_inputs = tf.concat([foregrounds, backgrounds], axis=-1)
                        fusion_inputs = tf.reshape(fusion_inputs, [-1,self.img_height,self.img_width,3*self.num_branch])
                        fusion_output = Fusion_forward(inputs=fusion_inputs, scope='Fusion/', training=False, reuse=tf.compat.v1.AUTO_REUSE)
                        self.traverse_results.append(out_mask)#(fusion_output) #B H W 1

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