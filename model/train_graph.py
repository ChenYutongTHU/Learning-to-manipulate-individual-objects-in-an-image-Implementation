import tensorflow as tf  
import os
from data import multi_texture_utils, flying_animals_utils#, multi_dsprites_utils, objects_room_utils
from .utils.generic_utils import bin_edge_map, train_op,myprint, myinput, erode_dilate, tf_resize_imgs, tf_normalize_imgs
from .utils.loss_utils import Generator_Loss, Inpainter_Loss, Supervised_Generator_Loss
from .nets import Generator_forward, Inpainter_forward, VAE_forward, Fusion_forward

mode2scopes = {
    'pretrain_inpainter': ['Inpainter'],
    'train_CIS': ['Inpainter','Generator'],
    'train_VAE': ['VAE//separate', 'VAE//fusion', 'Fusion'],
    'train_end2end': ['Inpainter','Generator','VAE//separate','VAE//fusion', 'Fusion']
}

class Train_Graph(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        self.batch_size = FLAGS.batch_size
        self.num_branch = FLAGS.num_branch
        self.img_height, self.img_width = FLAGS.img_height, FLAGS.img_width
    def build(self):
        self.is_training = tf.placeholder_with_default(True, shape=(), name="is_training")

        train_dataset, val_dataset = self.load_training_data(), self.load_val_data()
        self.train_iterator = train_dataset.make_one_shot_iterator()
        self.val_iterator = val_dataset.make_initializable_iterator()
        train_batch = self.train_iterator.get_next()
        
        current_batch = tf.cond(self.is_training, lambda: train_batch, lambda: self.val_iterator.get_next())

        self.image_batch, self.GT_masks = current_batch['img'], current_batch['masks']
        self.image_batch.set_shape([None, self.img_height, self.img_width, 3])
        self.GT_masks.set_shape([None, self.img_height, self.img_width, 1, None])
        if self.config.mode == 'pretrain_inpainter':
            self.generated_masks = self.Random_boxes()
        else:
            with tf.name_scope("Generator") as scope:
                self.generated_masks, generated_logits = Generator_forward(self.image_batch, self.config.dataset, 
                    self.num_branch, model=self.config.model, training=self.is_training, reuse=None, scope=scope)


        self.loss = {}
        with tf.name_scope("Inpainter") as scope:
            self.pred_intensities, self.unconditioned_mean, self.edge_map = \
                Inpainter_forward(self.num_branch, self.generated_masks, self.image_batch, dataset=self.config.dataset,
                    reuse=None, training=self.is_training, scope=scope)

        self.loss['Inpainter'], self.loss['Inpainter_branch'] = Inpainter_Loss(self.generated_masks, self.pred_intensities, self.image_batch)    
        self.loss['Generator'], self.loss['Generator_branch'], self.loss['Generator_denominator'], self.loss['Generator_numerator'] = \
            Generator_Loss(self.generated_masks, self.pred_intensities, self.image_batch, 
                    self.unconditioned_mean, self.config.epsilon)  
            
        if self.config.dataset == 'flying_animals':
            #normalize self.image_batch
            self.image_batch = tf_normalize_imgs(self.image_batch)
        #-----------VAE----------------
        if self.config.mode in ['train_VAE','train_end2end']:
            self.mask_capacity = tf.placeholder(shape=(), name='mask_capacity', dtype=tf.float32)
            with tf.name_scope("VAE") as scope:
                #-------------erode dilate  smooth------------
                filter_masks = []
                for i in range(self.config.num_branch):
                    filter_masks.append(erode_dilate(self.generated_masks[:,:,:,:,i]))
                self.generated_masks = tf.stack(filter_masks, axis=-1)#B H W 1 num_branch
                reordered = [self.generated_masks[:,:,:,:,i] for i in range(self.num_branch) if not i in self.config.bg] + \
                            [self.generated_masks[:,:,:,:,i] for i in self.config.bg]
                self.generated_masks = tf.stack(reordered, axis=-1) #B H W 1 M
                if self.config.dataset == 'flying_animals':
                    #resize image to (96, 128)
                    self.image_batch = tf_resize_imgs(self.image_batch, size=[self.img_height//2,self.img_width//2])
                    self.generated_masks = tf_resize_imgs(self.generated_masks, size=[self.img_height//2, self.img_width//2])

                VAE_loss, VAE_outputs = VAE_forward(image=self.image_batch, masks=self.generated_masks[:,:,:,:,:-1*len(self.config.bg)],  
                    bg_dim=self.config.bg_dim, tex_dim=self.config.tex_dim, mask_dim=self.config.mask_dim, 
                    scope=scope, reuse=None, training=self.is_training)


            self.loss['tex_kl'], self.loss['mask_kl'], self.loss['bg_kl'] = VAE_loss['tex_kl'], VAE_loss['mask_kl'], VAE_loss['bg_kl']
            self.loss['tex_kl_sum'], self.loss['bg_kl_sum'] = tf.reduce_sum(VAE_loss['tex_kl']), tf.reduce_sum(VAE_loss['bg_kl'])
            self.loss['mask_kl_sum'] = tf.abs(tf.reduce_sum(VAE_loss['mask_kl'])-self.mask_capacity)

            self.loss['tex_error'], self.loss['mask_error'], self.loss['bg_error'], self.loss['VAE_fusion_error'] = \
                VAE_loss['tex_error'], VAE_loss['mask_error'], VAE_loss['bg_error'], VAE_loss['fusion_error']


            self.loss['VAE//separate/texVAE'] = self.loss['tex_error']+self.config.tex_beta*self.loss['tex_kl_sum']
            self.loss['VAE//separate/maskVAE'] = self.loss['mask_error']+self.config.mask_gamma*self.loss['mask_kl_sum']
            self.loss['VAE//separate/bgVAE'] = self.loss['bg_error']+self.config.bg_beta*self.loss['bg_kl_sum']
            
            self.loss['VAE//separate'] = self.loss['VAE//separate/texVAE']+self.loss['VAE//separate/maskVAE']+self.loss['VAE//separate/bgVAE']
            self.loss['VAE//fusion'] = self.loss['VAE_fusion_error']



            #-----------fuse all branch---------------
            foregrounds = VAE_outputs['out_fusion'] #B H W 3 (num_branch-1)
            background_mask = 1-tf.reduce_sum(VAE_outputs['out_masks'], axis=-1, keepdims=True) #B H W 1 1
            backgrounds = tf.expand_dims(VAE_outputs['out_bg'], axis=-1)*background_mask # B H W 3 1

            fusion_inputs = tf.concat([foregrounds, backgrounds], axis=-1) #B H W 3 fg_branch+1
            fusion_inputs = tf.reshape(fusion_inputs, [-1,fusion_inputs.get_shape()[1],fusion_inputs.get_shape()[2],3*fusion_inputs.get_shape()[4]])

            with tf.name_scope("Fusion") as scope: 
                self.fusion_outputs = Fusion_forward(inputs=fusion_inputs, scope=scope, training=self.is_training, reuse=None)
            Fusion_error = tf.abs(self.fusion_outputs-self.image_batch) #  B H W 3
            self.loss['Fusion'] = tf.reduce_mean(tf.reduce_sum(Fusion_error, axis=[1,2,3]))  #B -> ,(average on batch)

            self.loss['CIS'] = self.loss['Generator']
            self.loss['Generator'] = (self.loss['tex_error']+self.loss['mask_error'])*self.config.VAE_weight + \
                self.loss['CIS']*self.config.CIS_weight

            self.VAE_outtexes, self.VAE_outtex_bg = VAE_outputs['out_texes'], VAE_outputs['out_bg']
            self.VAE_outmasks = VAE_outputs['out_masks']  #no background  #B H W (num_branch-1)
            self.VAE_fusion = VAE_outputs['out_fusion']

            if self.config.dataset == 'flying_animals':
                #resize VAE_out
                import functools
                resize = functools.partial(tf_resize_imgs, size=[self.img_height, self.img_width])
                self.generated_masks, self.image_batch = resize(self.generated_masks), resize(self.image_batch)
                self.VAE_outtexes, self.VAE_outtex_bg, self.VAE_outmasks = resize(self.VAE_outtexes), resize(self.VAE_outtex_bg), resize(self.VAE_outmasks)
                self.VAE_fusion, self.fusion_outputs = resize(self.VAE_fusion), resize(self.fusion_outputs)



        #------------------------------
        with tf.name_scope('train_op'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.vae_global_step = tf.Variable(0, name='vae_global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)
            self.incr_vae_global_step = tf.assign(self.vae_global_step, self.vae_global_step+1)
            self.train_ops, self.train_vars_grads = self.get_train_ops_grads()

        with tf.name_scope('summary_vars'):
            self.loss['Generator_var'] = tf.Variable(0.0, name='Generator_var')
            self.loss['Generator_branch_var'] = tf.Variable(0.0, name='Generator_branch_var')
            self.loss['Inpainter_branch_var'] = tf.Variable(0.0, name='Inpainter_branch_var')  # do tf.summary
            self.loss['Generator_denominator_var'] = tf.Variable(0.0, name='Generator_denominator_var')
            self.loss['EvalIoU_var'] = tf.Variable(0.0, name='EvalIoU_var')
            if self.config.mode in ['train_VAE','train_end2end']:
                self.loss['tex_kl_var'] = tf.Variable(0.0, name='tex_kl_var') 
                self.loss['mask_kl_var'] = tf.Variable(0.0, name='mask_kl_var') 
                self.loss['bg_kl_var'] = tf.Variable(0.0, name='bg_kl_var') 
        #   
    def get_train_ops_grads(self):
        #generate train_ops 
        lr_dict = {'Generator':self.config.gen_lr, 'Inpainter':self.config.inp_lr, 
            'VAE//separate':self.config.VAE_lr, 'VAE//fusion':self.config.VAE_lr, 'Fusion':self.config.VAE_lr}
        scopes = mode2scopes[self.config.mode]
        train_ops, train_vars_grads = {}, {}
        for m in scopes:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_dict[m])
            train_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, m)
            update_op = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, m)
            train_ops[m], train_vars_grads[m] = train_op(loss=self.loss[m],var_list=train_vars, optimizer=optimizer)
            train_ops[m] = tf.group([train_ops[m], update_op])
        return  train_ops, train_vars_grads

    def load_training_data(self):
        if self.config.dataset == 'multi_texture':
            return multi_texture_utils.dataset(self.config.root_dir, phase='train',
                batch_size=self.batch_size, max_num=self.config.max_num)
        elif self.config.dataset == 'flying_animals':
            return flying_animals_utils.dataset(self.config.root_dir, phase='train',
                batch_size=self.batch_size, max_num=self.config.max_num)

        elif self.config.dataset == 'multi_dsprites':
            return multi_dsprites_utils.dataset(self.config.root_dir, phase='train',
                batch_size=self.batch_size, skipnum=self.config.skipnum, takenum=self.config.takenum,
                shuffle=True, map_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif self.config.dataset == 'objects_room':
            return objects_room_utils.dataset(self.config.root_dir, phase='train',
                batch_size=self.batch_size, skipnum=self.config.skipnum, takenum=self.config.takenum,
                shuffle=True, map_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            raise IOError("Unknown Dataset")
                # B H W 3
    def load_val_data(self):
        if self.config.dataset == 'multi_texture':
            return multi_texture_utils.dataset(self.config.root_dir, phase='val',
                batch_size=8, max_num=self.config.max_num)
        elif self.config.dataset == 'flying_animals':
            return flying_animals_utils.dataset(self.config.root_dir, phase='val',
                batch_size=8, max_num=self.config.max_num)

        elif self.config.dataset == 'multi_dsprites':
            return multi_dsprites_utils.dataset(self.config.root_dir, phase='val',
                batch_size=self.batch_size, skipnum=0, takenum=self.config.skipnum,
                shuffle=False, map_parallel_calls=1)     
        elif self.config.dataset == 'objects_room':
            return objects_room_utils.dataset(self.config.root_dir, phase='val',
                batch_size=self.batch_size, skipnum=0, takenum=self.config.skipnum,
                shuffle=False, map_parallel_calls=1)       
        else:
            raise IOError("Unknown Dataset")

    def Random_boxes(self):
        b,h,w = self.batch_size, self.config.img_height, self.config.img_width
        r1 = tf.random.uniform(shape=[], minval=0, maxval=h*2//3, dtype=tf.int32)
        r2 = tf.random.uniform(shape=[], minval=r1+h//5, maxval=h-1, dtype=tf.int32)
        c1 = tf.random.uniform(shape=[], minval=0, maxval=w*2//3, dtype=tf.int32)
        c2 = tf.random.uniform(shape=[], minval=c1+w//5, maxval=w-1, dtype=tf.int32)
        ones = tf.ones([b,h,w,1])
        zeros = tf.zeros([b,h,w,1])
        random_box = tf.concat([zeros[:,0:r1,:,:],ones[:,r1:r2,:,:],zeros[:,r2:,:,:]], axis=1)
        random_box = tf.concat([zeros[:,:,0:c1,:],random_box[:,:,c1:c2,:],zeros[:,:,c2:,:]], axis=2)
        random_boxes = tf.stack([random_box, 1-random_box], axis=-1)
        #B H W 1
        return random_boxes #B H W 1 2
