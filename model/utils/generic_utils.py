import numpy as np
import cv2
import tensorflow as tf
import random
import os
import imageio
import math


def erode_dilate(img):
    img = -1*tf.nn.max_pool2d(-1*img, ksize=3, strides=1, padding='SAME')
    img = tf.nn.max_pool2d(img, ksize=3, strides=1, padding='SAME')
    return img

def tf_normalize_imgs(imgs):
    #normalize to 0~1
    #B H W 3
    max_value = tf.reduce_max(imgs, axis=[1,2], keepdims=True)
    min_value = tf.reduce_min(imgs, axis=[1,2], keepdims=True)
    imgs = tf.math.divide_no_nan(imgs-min_value, max_value-min_value)
    return imgs

def tf_resize_imgs(imgs, size):
    #default bilinear
    shape = imgs.get_shape().as_list()
    assert len(shape) <=5 and len(shape)>=3
    if len(shape) <= 4: #B H W C
        return tf.image.resize(imgs, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    else: #B H W C M
        resize_imgs = [tf.image.resize(imgs[:,:,:,:,i], size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) for i in range(shape[-1])]
        resize_imgs = tf.stack(resize_imgs, axis=-1)
        return resize_imgs

def myprint(string):
    print ("\033[0;30;42m"+string+"\033[0m")

def myinput(string=""):
    return input ("\033[0;30;41m"+string+"\033[0m")

def bin_edge_map(imgs, dataset):
    assert dataset in ['flying_animals', 'multi_texture', 'multi_dsprites', 'objects_room']
    if dataset == 'flying_animals':
        d_xy = tf.reduce_sum(tf.image.sobel_edges(imgs), axis=-2) 
        d_xy = d_xy/15
        d_xy = tf.cast(tf.abs(d_xy)>0.04, tf.float32)
    else:
        d_xy = tf.abs(tf.image.sobel_edges(imgs)) # B H W 3 2
        d_xy = tf.reduce_sum(d_xy,axis=-2, keepdims=False) #B H W 2
        #adjust the range to 0~1
        max_val = tf.math.reduce_max(d_xy, axis=[1,2], keepdims=True)
        min_val = tf.math.reduce_min(d_xy, axis=[1,2], keepdims=True)
        d_xy = tf.math.divide_no_nan(d_xy-min_val,max_val-min_val) #0~1
        #sparse binary edge map 
        threshold = {'multi_texture': 0.5, 'multi_dsprites':0.0001, 'objects_room':0.2}
        d_xy = tf.cast(d_xy>threshold[dataset], tf.float32)
    return d_xy

def Permute_IoU(label, pred):
    A, B = label, pred
    #H W 1 C ndarray
    H,W,nc,N = A.shape
    #not a real permutation
    ans = 0
    nc = 0 #non-empty channel

    ans_perfg = []
    for i in range(N):
        src = np.expand_dims(A[:,:,:,i], axis=-1)>0.04 #H W 1 1 #binary
        if np.sum(src) > 4:
            nc += 1
            trg = B>0.04  #H W 1 C
            U = np.sum(src+trg, axis=(0,1,2))  #H W 1 C  -> C
            I = np.sum(src*trg, axis=(0,1,2))  #H W 1 C -> C
            eps = 1e-8
            IoU = I/(eps+U)
            ans += np.max(IoU)
            ans_perfg.append(np.max(IoU))
        else:
            ans_perfg.append(0)
    assert nc >0
    return ans/nc#, ans_perfg  #N,

def train_op(loss, var_list, optimizer, gradient_clip_value=-1):
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    
    clipped_grad_and_vars = [ (ClipIfNotNone(grad, gradient_clip_value),var) \
                              for grad, var in grads_and_vars ]
    train_operation = optimizer.apply_gradients(clipped_grad_and_vars)

    return train_operation, clipped_grad_and_vars

def ClipIfNotNone(grad, clipvalue):
    if clipvalue==-1:
        return grad
    else:
        return tf.clip_by_value(grad, -clipvalue, clipvalue)