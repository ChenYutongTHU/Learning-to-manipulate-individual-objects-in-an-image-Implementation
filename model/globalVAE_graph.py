import tensorflow as tf  
import os
from data import multi_texture_utils, flying_animals_utils, multi_dsprites_utils, objects_room_utils
from .utils.generic_utils import bin_edge_map, train_op,myprint, myinput, erode_dilate, tf_resize_imgs, tf_normalize_imgs
from .utils.loss_utils import Generator_Loss, Inpainter_Loss, Supervised_Generator_Loss
from .nets import Generator_forward, Inpainter_forward, VAE_forward, Fusion_forward, encoder_decoder, gaussian_kl


class Train_Graph(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        #load data
        self.batch_size = FLAGS.batch_size
        self.img_height, self.img_width = FLAGS.img_height, FLAGS.img_width
        #hyperparameters
    def build(self):
        train_dataset = self.load_training_data()
        self.train_iterator = train_dataset.make_one_shot_iterator()
        train_batch = self.train_iterator.get_next()

        self.image_batch, self.GT_masks = train_batch['img'], train_batch['masks']
        self.image_batch.set_shape([None, self.img_height, self.img_width, 3])
       
        with tf.compat.v1.variable_scope("VAE") as scope:
            z_mean, z_log_sigma_sq, out_logit = encoder_decoder(x=self.image_batch, output_ch=3, latent_dim=self.config.tex_dim, training=True)            
        
        self.latent_loss_dim = tf.reduce_mean(gaussian_kl(z_mean, z_log_sigma_sq), 0) #average on batch  dim,
        self.latent_loss = tf.reduce_sum(self.latent_loss_dim)


        self.out_imgs = tf.nn.sigmoid(out_logit)  #0~1

        if self.config.VAE_loss == 'L1':
            self.reconstr_loss = tf.reduce_sum(tf.reduce_mean(tf.abs(self.image_batch-self.out_imgs), axis=0))  #B H W 3
        else:
            self.reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.image_batch, logits=out_logit) #B H W 3
            self.reconstr_loss = tf.reduce_sum(self.reconstr_loss, axis=[1,2,3]) #B,
            self.reconstr_loss = tf.reduce_mean(self.reconstr_loss) 

        self.loss = self.reconstr_loss+self.config.tex_beta*self.latent_loss



        #------------------------------
        with tf.name_scope('train_op'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)
            self.train_ops, self.train_vars_grads = self.get_train_ops_grads()

        with tf.name_scope('summary_vars'):
            self.kl_var = tf.Variable(0.0, name='kl_var') 
        #   
    def get_train_ops_grads(self):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config.VAE_lr)
        train_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'VAE')
        update_op = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, 'VAE')
        train_ops, train_vars_grads = train_op(loss=self.loss,
                var_list=train_vars, optimizer=optimizer, gradient_clip_value=-1)
        train_ops = tf.group([train_ops, update_op])
        return  train_ops, train_vars_grads

    def load_training_data(self):
        if self.config.dataset == 'multi_texture':
            return multi_texture_utils.dataset(self.config.root_dir, val=False,
                batch_size=self.batch_size, max_num=self.config.max_num, 
                zoom=('z' in self.config.variant), 
                rotation=('r' in self.config.variant), 
                texture_transform=self.config.texture_transform)
        elif self.config.dataset == 'flying_animals':
            return flying_animals_utils.dataset(self.config.root_dir,val=False,
                batch_size=self.batch_size, max_num=self.config.max_num)
        elif self.config.dataset == 'multi_dsprites':
            return multi_dsprites_utils.dataset(self.config.root_dir,val=False,
                batch_size=self.batch_size, skipnum=self.config.skipnum, takenum=self.config.takenum,
                shuffle=True, map_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif self.config.dataset == 'objects_room':
            return objects_room_utils.dataset(self.config.root_dir,val=False,
                batch_size=self.batch_size, skipnum=self.config.skipnum, takenum=self.config.takenum,
                shuffle=True, map_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            raise IOError("Unknown Dataset")
