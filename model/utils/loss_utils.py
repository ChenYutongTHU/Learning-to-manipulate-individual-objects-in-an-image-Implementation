from __future__ import division
import tensorflow as tf

import tensorflow as tf

def region_error(X, Y, region):
    #A  B H W c / B H W 3 M
    #B B H W 3 / B H W 3 M
    #region B H W 1 /B H W 1 M
    X = tf.cond(tf.equal(tf.rank(X), 5), lambda:X, lambda:tf.expand_dims(X, axis=-1))
    Y = tf.cond(tf.equal(tf.rank(Y), 5), lambda:Y, lambda:tf.expand_dims(Y, axis=-1))
    error = tf.abs(X-Y)*region  #B H W 3 M
    error = tf.reduce_sum(error, axis=[1,2,3]) #B,M
    return error #B,M

def Supervised_Generator_Loss(pred, GT):
    #B H W 1 C
    # use cross entropy
    error = tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=GT, logits=pred, dim=-1)#, axis=-1)#B H W 1  #???
    error = tf.reduce_mean(error) #B H W 1 -> perpixel 
    return error

def Generator_Loss(masks, pred_intensities, image, unconditioned_mean, epsilon):
    #masks B H W 1 C
    #pred_intensities B H W 3 C
    #unconditioned_mean B H W 3
    #image B H W 3
    numerator = region_error(pred_intensities, image, masks) #B,C
    denominator = epsilon + region_error(unconditioned_mean, image, masks) # B,C
    IRR = 1-tf.math.divide(numerator, denominator) #B,C
    perbranch_loss = tf.reduce_mean(IRR, axis=0) #C, 
    loss = tf.reduce_sum(perbranch_loss)
    return loss, IRR, denominator, numerator #scalar, 

def Inpainter_Loss(masks, pred_intensities, image):
    #pred_intensities 0~1
    #image 0~1
    B, H, W, C = image.get_shape().as_list() #B H W C
    num_pixel = H*W*C
    loss0 = region_error(pred_intensities, image, masks) #B,M
    perbranch_loss = tf.reduce_mean(loss0, axis=0) # M,  
    loss = tf.reduce_sum(perbranch_loss) #M->scalar
    loss = tf.math.divide(loss, num_pixel)
    return loss, loss0