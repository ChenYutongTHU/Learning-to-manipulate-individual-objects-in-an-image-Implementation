import tensorflow as tf 
import os
import gflags  
#https://github.com/google/python-gflags
import sys
sys.path.append("..") 
import pprint
from keras.utils.generic_utils import Progbar
import model.Summary as Summary
from model.utils.generic_utils import myprint, myinput, bin_edge_map
from data import flying_animals_utils
from model.eval_graph import Eval_Graph
from model.nets import inpaint_net
import imageio
import numpy as np
import time
import random
# for evaluate CIS term

def convert(img):
    #float32 0~1 -> uint8 0~255
    return (img*255).astype(np.uint8)

tf.compat.v1.set_random_seed(479)
random.seed(101)

dataset = flying_animals_utils.dataset('../data/flying_animals_data/img_data.npz', 1, count=5)
iterator = dataset.make_one_shot_iterator()
batch = iterator.get_next()
img_batch = batch['img']
edge_map = bin_edge_map(img_batch)
with tf.name_scope("Inpainter") as scope:
    unconditioned_mean = inpaint_net(tf.zeros_like(img_batch), tf.ones_like(img_batch[:,:,:,0:1]),   #B H W 3
        edge_map, scope=scope, reuse=False, training=True)

sess = tf.Session()
op = 'debug'


#restore
path = '/home/yutong/Object-Centric-Representation/save_checkpoint/kif_denseedge_recover/recover_net-60000'
# feed forward img_batch-0.5  (-0.5~0.5)
# output mean = mean + 0.5

#path = '/media/DATA2_6TB/yutong/learning2manip/fa2_recover/pretrain_inpainter/c840d_bs8_lr1e-4/Inpainter_Sum/Inpainter-33214'
var_dict = dict()
for v in tf.trainable_variables('Inpainter'):
    #Inpainter// ->InpaintNet
    newname = 'InpaintNet'+v.op.name[9:]
    var_dict[newname] = v
print(var_dict)
saver = tf.train.Saver(var_dict)
saver.restore(sess, path)
for i in range(10):
    (img, mean) = sess.run((img_batch, unconditioned_mean))
    mean = mean + 0.5 
    #1 H W 3 1 H W 3
    imageio.imwrite(op+'/img{}.png'.format(i), (img[0]*255).astype(np.uint8))
    imageio.imwrite(op+'/img_pred{}.png'.format(i), (mean[0]*255).astype(np.uint8))