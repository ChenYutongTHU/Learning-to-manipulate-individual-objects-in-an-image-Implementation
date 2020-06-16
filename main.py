import tensorflow as tf 
from itertools import count
import os
import gflags  
from git import Repo
import sys
import pprint 
from common_flags import FLAGS
import warnings
from trainer import train_inpainter, train_CIS, train_VAE
from eval import eval_VAE
from model.utils.generic_utils import myprint, myinput
import random
import numpy as np

def save_log(source, trg_dir, print_flags_dict, sha):
    file_name = source.split('/')[-1]
    new_file = os.path.join(trg_dir, file_name)
    log_name = 'log'
    while os.path.isfile(new_file):
        new_file =  new_file[:-3]+'_c.sh' #.sh
        log_name += '_c'
    os.system('cp '+source+' '+ new_file)
    myprint ("Save "+source +" as "+new_file)
    log_file = os.path.join(trg_dir, log_name+'.txt')
    with open(log_file,'w') as log_stream:
        log_stream.write('commit:' + sha + '\n')
        pprint.pprint(print_flags_dict, log_stream)
    with open(new_file, 'a') as sh_stream:
        sh_stream.write('\n#commit:'+sha)
    myprint('Corresponding log file '+log_file)
    myinput("Enter to continue")
    os.system('chmod a=rx '+log_file)
    os.system('chmod a=rx '+new_file)
    return 

def complete_FLAGS(FLAGS):
    #complete some configuration given the speficied dataset

    img_size_dict = {'multi_texture': (64,64),
            'multi_dsprites': (64,64),
            'objects_room': (64,64),
            'flying_animals': (192,256)}
    max_num_dict = {'multi_texture':4,
        'objects_room': 5,
        'multi_dsprites':4,
        'flying_animals':5}
    FLAGS.img_height, FLAGS.img_width = img_size_dict[FLAGS.dataset]
    FLAGS.max_num = max_num_dict[FLAGS.dataset]
    if FLAGS.PC and FLAGS.dataset=='multi_texture':
        FLAGS.max_num=2

    if FLAGS.mode == 'pretrain_inpainter':
        FLAGS.num_branch = 2
    else:
        assert FLAGS.num_branch >= FLAGS.max_num+1

    FLAGS.n_bg = 3 if FLAGS.dataset=='objects_room' else 1
    return 

def main(argv):
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        print ('FlagsError: ', e)
        sys.exit(1)
    else:
        tf.compat.v1.set_random_seed(479)
        random.seed(101)
        complete_FLAGS(FLAGS) 
        pp = pprint.PrettyPrinter()
        print_flags_dict = {}
        for key in FLAGS.__flags.keys():
            print_flags_dict[key] = getattr(FLAGS, key)
        pp.pprint(print_flags_dict)
        myinput("Press enter to continue")
       
        repo = Repo()
        sha = repo.head.object.hexsha
        FLAGS.checkpoint_dir = FLAGS.checkpoint_dir[:-1] if FLAGS.checkpoint_dir[-1]=='/' else FLAGS.checkpoint_dir
        if os.path.exists(FLAGS.checkpoint_dir):
            I = myinput(FLAGS.checkpoint_dir+' already exists. \n Are you sure to'
                ' place the outputs in the same dir?  Y or Y! or N\n' 
                'Y: resume training, save previous outputs in the dir and continue saving outputs in it\n'
                'Y!: restart training, delete previous outputs in the dir\n'
                'N to quit \n')
            if I in ['Y','y']:
                save_log(FLAGS.sh_path, FLAGS.checkpoint_dir, print_flags_dict, sha)
                import time
                tf.compat.v1.set_random_seed(time.localtime()[5]*10)
                random.seed(time.localtime()[4]*10)  #new random seed
            elif I in ['N','n']:
                sys.exit(1)
            else:
                os.system('rm -f -r '+FLAGS.checkpoint_dir+'/*')
                save_log(FLAGS.sh_path, FLAGS.checkpoint_dir, print_flags_dict, sha)                
        else:
            os.makedirs(FLAGS.checkpoint_dir)
            save_log(FLAGS.sh_path, FLAGS.checkpoint_dir, print_flags_dict, sha)

        
        assert FLAGS.mode in ['pretrain_inpainter','train_CIS', 'train_VAE', 'train_end2end'
            'eval_CIS', 'eval_VAE']
        
        if FLAGS.mode == 'pretrain_inpainter':
            train_inpainter.train(FLAGS)
        elif FLAGS.mode == 'train_CIS':
            train_CIS.train(FLAGS)
        elif FLAGS.mode == 'eval_VAE':
            eval_VAE.eval(FLAGS)
        # elif FLAGS.mode == 'train_end2end':
        #     train_end2end.train(FLAGS)
        elif FLAGS.mode == 'train_VAE':
            train_VAE.train(FLAGS)
        else:
            pass
        #     pass
        # elif FLAGS.mode == 'eval_CIS':
        #     eval_CIS.eval(FLAGS)

if __name__ == '__main__':
    main(sys.argv)
