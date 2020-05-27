import gflags
FLAGS = gflags.FLAGS



#data
gflags.DEFINE_string('dataset', 'multi_dsprites/ multi_texture/ objects_room/ flying_animals', 'Dataset used')
gflags.DEFINE_string('root_dir',"/your/path/to/dataset", 'Folder containig the dataset')


gflags.DEFINE_integer("takenum", -1, 'take number  default: the entire dataset. not used for flying_animals')
gflags.DEFINE_integer('skipnum',2000,'skip number default: 2k used for testset not used for flying_animals')
gflags.DEFINE_bool('shuffle',False,'')


#dir
gflags.DEFINE_string('checkpoint_dir', "", "Experiment folder. It will contain"
                     "the saved checkpoints, tensorboard logs or evaluation results.")
#gflags.DEFINE_string('output_dir',"./outputs/0","Containing outputs result when doing evaluation")
gflags.DEFINE_integer('summaries_secs', 40, 'number of seconds between computation of summaries, used in train_inpainter')
gflags.DEFINE_integer('summaries_steps', 100, 'number of step between computation of summaries, used in train_CIS')
gflags.DEFINE_integer('ckpt_secs', 3600, 'number of seconds between checkpoint saving')
gflags.DEFINE_integer('ckpt_steps', 10000, 'number of step between checkpoint saving')


#resume
gflags.DEFINE_bool('resume_fullmodel', False, 'whether to resume a fullmodel')
gflags.DEFINE_bool('resume_inpainter', True, 'resume pretrained inpainter for train_CIS  inpainter_ckpt needed')
gflags.DEFINE_bool('resume_resnet', False, 'whether to use pretrained resnet (effective when resume_fullmodel=False)')
gflags.DEFINE_bool('resume_CIS', False, 'whether to resume inpainter and generator')
#checkpoint to load 
# used for resumed training or evaluation
gflags.DEFINE_string('fullmodel_ckpt', '?', 'checkpoint of full model  inpainter+Generator(train_CIS) inp+gen+VAE(train_end2end)')
gflags.DEFINE_string('CIS_ckpt', '?', 'checkpoint of inpainter + generator')
gflags.DEFINE_string('mask_ckpt', '?', '')
gflags.DEFINE_string('tex_ckpt', '?', '')
gflags.DEFINE_string('generator_ckpt', '?', 'checkpoint of mask generator')
gflags.DEFINE_string('inpainter_ckpt', '?', 'checkpoint of pretrained inpainter')
gflags.DEFINE_string('resnet_ckpt', 'resnet/resnet_v2_50.ckpt', 'checkpoint of pretrained resnet')
#to - do  VAE (TEXTURE AND SHAPE)



gflags.DEFINE_integer('max_training_hrs', 72,'maximum training hours')
#copy the sh

#mode
gflags.DEFINE_string('mode', 'train_CIS', 'pretrain_inpainter / train_CIS / train_VAE / eval_segment / eval_VAE /train_supGenerator')
gflags.DEFINE_string('sh_path','./train.sh', 'absolute path of the running shell')



#
gflags.DEFINE_integer('batch_size', 32, 'batch_size')
gflags.DEFINE_integer('num_branch', 6, 'output channel of segmentation') 
gflags.DEFINE_integer('nobj', -1, 'number of objects, only used in evaluation or fixed_number training')

#network
gflags.DEFINE_string('model', 'resnet_v2_50', 'resnet_v2_50 or resnet_v2_101 or segnet')
#VAE
gflags.DEFINE_integer('tex_dim', 4, 'dimension of texture latent space') 
gflags.DEFINE_integer('mask_dim', 10, 'dimension of mask latent space')
gflags.DEFINE_integer('bg_dim', 10, 'dimension of bg latent space')
gflags.DEFINE_float('VAE_weight', 0,'weight of tex_error and mask_error loss for Generator when training End2End')
gflags.DEFINE_float('CIS_weight', 1,'weight of CIS loss for Generator when training End2End')
gflags.DEFINE_float('tex_beta', 10,'ratio of tex_error loss and tex_kl loss')
gflags.DEFINE_float('mask_gamma', 50000,'')
gflags.DEFINE_float('mask_capacity_inc',1e-5, 'increment of mask capacity at each step')
gflags.DEFINE_float('bg_beta', 10,'ratio of bg_error loss and bg_kl loss')

#hyperparameters
gflags.DEFINE_float('gen_lr',1e-3,'learning rate')
gflags.DEFINE_float('inp_lr',1e-4,'learning rate')
gflags.DEFINE_float('VAE_lr',1e-4,'learning rate')
gflags.DEFINE_float('epsilon', 40, 'epsilon in the denominator of IRR')
gflags.DEFINE_float('gen_clip_value', -1, 'generator''s grad_clip_value -1 means no clip')
gflags.DEFINE_integer('iters_inp', 1, 'iteration # of inpainter')
gflags.DEFINE_integer('iters_gen', 3, 'iteration # of generator')
gflags.DEFINE_integer('iters_gen_vae', 3, 'iteration # of generator and vae  used when training end2end')


#flying animals (only support)
gflags.DEFINE_integer('max_num',5,'max number of objects in the image')
#gflags.DEFINE_integer('min_num',1,'min number of objects in the image')
gflags.DEFINE_integer('bg_num', 100, 'number of bg')
gflags.DEFINE_integer('ani_num',240,'')



#automatically set flags
gflags.DEFINE_integer('img_height',64,'')
gflags.DEFINE_integer('img_width',64,'')
gflags.DEFINE_integer('n_bg',1,'')


#traverse
gflags.DEFINE_string('input_img','./?','input image path')
gflags.DEFINE_string('traverse_type', 'tex', 'tex or branch')
gflags.DEFINE_string('traverse_dim', 'all', 'all or #1,#2,#3')
gflags.DEFINE_string('traverse_branch', 'all', 'all or #1,#2,#3')
gflags.DEFINE_float('traverse_range', '5', 'k z_mean +- k*sigma')


gflags.DEFINE_string('VAE_loss','CE','CE or L1')