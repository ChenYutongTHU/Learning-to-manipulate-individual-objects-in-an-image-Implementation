import tensorflow as tf 
import os
import sys
sys.path.append("..") 
import pprint
from model.utils.generic_utils import myprint, myinput, Permute_IoU, reorder_mask
from model.nets import Generator_forward
import imageio
import numpy as np
import time
from data import multi_texture_utils, flying_animals_utils, objects_room_utils, multi_dsprites_utils
import argparse

parser = argparse.ArgumentParser(description='test the segmentation mean IoU')
parser.add_argument('--data_path', type=str, help='dir of the test set')
parser.add_argument('--batch_size', default=8, type=int, help='batchsize', required=False)
parser.add_argument('--dataset_name', default='flying_animals', type=str, help='flying_animals / multi_texture / multi_dsprites / objects_room')
parser.add_argument('--ckpt_path', default='./', type=str)
parser.add_argument('--num_branch', default=6, type=int, help='#branch should match the checkpoint and network')
parser.add_argument('--PC',default=False,type=bool, help='whether to test perceptual consistency, output identity switching rate')
parser.add_argument('--model',default='resnet_v2_50',type=str, help='segmentation network model, resnet_v2_50 or segnet')
args = parser.parse_args()

#usage python test/test_segmentation.py  arg1 arg2 arg3


data_path = args.data_path
batch_size = args.batch_size
dataset_name = args.dataset_name
num_branch = args.num_branch
PC, model = args.PC, args.model

if dataset_name == 'flying_animals':
    img_height, img_width = 192, 256
    dataset = flying_animals_utils.dataset(data_path=data_path,batch_size=batch_size, max_num=5, phase='test')
elif dataset_name == 'multi_texture':
    img_height, img_width = 64, 64
    dataset = multi_texture_utils.dataset(data_path=data_path, batch_size=batch_size, max_num=2 if PC else 4, phase='test',PC=PC)
elif dataset_name == 'multi_dsprites':
    img_height, img_width = 64, 64
    dataset = multi_dsprites_utils.dataset(tfrecords_path=data_path,batch_size=batch_size, phase='test')   
elif dataset_name == 'objects_room':
    img_height, img_width = 64, 64
    dataset = objects_room_utils.dataset(tfrecords_path=data_path,batch_size=batch_size, phase='test')      

ckpt_path = args.ckpt_path


iterator = dataset.make_one_shot_iterator()
test_batch = iterator.get_next()
img, tf_GT_masks = test_batch['img'], test_batch['masks']
img.set_shape([batch_size, img_height, img_width, 3])
tf_GT_masks.set_shape([batch_size, img_height, img_width, 1, None])

with tf.name_scope("Generator") as scope:
    tf_generated_masks, null = Generator_forward(img, dataset_name, 
        num_branch, model=model, training=False, reuse=None, scope=scope)
    tf_generated_masks = reorder_mask(tf_generated_masks) #place the background at the last channel

restore_vars = tf.global_variables('Generator')
saver = tf.train.Saver(restore_vars)

with tf.Session() as sess:
    saver.restore(sess, ckpt_path)
    scores = []
    fetches = {'GT_masks': tf_GT_masks, 'generated_masks': tf_generated_masks, 'img':img}

    if PC:
        #test perceptual consistency
        num = 9*9*9*9-1
        assert num%batch_size==0
        niter = num//batch_size
        score, arg_maxIoUs = 0, []
        for u in range(niter):
            results = sess.run(fetches)
            for j in range(batch_size):
                s, arg_maxIoU = Permute_IoU(label=results['GT_masks'][j], pred=results['generated_masks'][j])
                score += s
                arg_maxIoUs.append(arg_maxIoU)
        score = score/num
        arg_maxIoUs = np.stack(arg_maxIoUs, axis=0) 
        count = np.sum(arg_maxIoUs, axis=0) 
        switching_rate = np.min(count)/num
        print("IoU: {} identity switching rate: {} ".format(score, switching_rate))


    else:
        for i in range(10):  #10 subsets
            #200 images in each subset
            assert 200%batch_size==0
            niter = 200//batch_size
            score = []
            for u in range(niter):
                results = sess.run(fetches)
                for j in range(batch_size):
                    s, null = Permute_IoU(label=results['GT_masks'][j], pred=results['generated_masks'][j])
                    score.append(s)
            scores.append(score) #10*200
            print("subset {}: mean {} variance{}\n".format(i+1, np.mean(scores[i]), np.var(scores[i])))
        mean_IoU = np.mean(scores, axis=-1) #10,
        print("mean of mean_IoU: {}  std of mean_IoU: {}\n".format(np.mean(mean_IoU), np.std(mean_IoU, ddof=1)))    
 
