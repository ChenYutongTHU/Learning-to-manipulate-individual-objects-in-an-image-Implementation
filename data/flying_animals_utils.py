import tensorflow as tf
import numpy as np
import random
import imageio
import os
import functools
import scipy.ndimage as ndimage

#online generator for flying animals dataset 

H, W = 192,256
def convert(imgs):
    return (imgs/255).astype(np.float32)

def check_occlusion(pos, pos_list):
    for p in pos_list:
        dist = abs(pos[0]-p[0])+abs(pos[1]-p[1])
        if dist<=97:
            return True
    return False

def generate_params(data, max_num, num):
    deter_params = []
    bg_num = data['background'].shape[0]-1 #100
    ani_num = data['foreground'].shape[0]-1 #240
    for i in range(num):  #size of the validation set
        param = dict()
        param['bg_index'], param['number'] = random.randint(1, bg_num), get_number(random.uniform(0,1))
        param['fg_indices'] = [random.randint(1,ani_num) for k in range(param['number'])] +[0]*(max_num-param['number'])
        
        pos_list = []  
        params = []
        for k in range(param['number']): 
            f = random.uniform(0.9, 1.2)   
            dx, dy = random.uniform(-W*2.2//5, W*2.2//5), random.uniform(-H*2.2//5, H*2.2//5)
            while check_occlusion([dx, dy], pos_list):
                dx, dy = random.uniform(-W*2.2//5, W*2.2//5), random.uniform(-H*2.2//5, H*2.2//5)
            pos_list.append([dx,dy])
            p= [f,0,W/2-f*W/2-f*dx,0,f,H/2-f*H/2-f*dy,0,0]
            params.append(p)
        params += [[1,0,0,0,1,0,0,0]]*(max_num-param['number'])
        param['params'] = params  #factor translation 
        deter_params.append(param)
    return deter_params  # list of dicts 5*100 element

def get_number(n):
    #probability of distribution -- number of animals in an image
    if n<=0.8:
        return 5  #0.8
    elif n<=0.9:
        return 4  #0.1
    elif n<=0.96:
        return 3  #0.06
    elif n<=0.99:
        return 2  #0.03
    else:
        return 1 #0.01

def params_gen(data, max_num, deterministic_params=None):
    backgrounds = convert(data['background'])
    foregrounds = convert(data['foreground'])
    ani_masks = data['mask'].astype(np.float32)

    bg_num = data['background'].shape[0]-1 #80 or 100
    ani_num = data['foreground'].shape[0]-1 #240

    step = 0
    while True:
        step += 1
        if deterministic_params:  #200 images for validation
            param = deterministic_params[(step-1)%len(deterministic_params)]
            bg = backgrounds[param['bg_index']]
            texes = [foregrounds[i] for i in param['fg_indices']]
            masks = [ani_masks[i] for i in param['fg_indices']]
            yield bg, np.stack(texes, axis=0), np.stack(masks, axis=0), param['params']
        else: #online generated data (infinite)
            number = get_number(random.uniform(0,1))           
            params = []
            pos_list = []
            texes = []
            masks = []
            bg = backgrounds[random.randint(1,bg_num)]
            for i in range(number):
                ind = random.randint(1,ani_num)
                texes.append(foregrounds[ind])
                masks.append(ani_masks[ind])
                f = random.uniform(0.9, 1.2)  #input /output
                dx, dy = random.uniform(-W*2.2//5, W*2.2//5), random.uniform(-H*2.2//5, H*2.2//5)
                while check_occlusion([dx, dy], pos_list):
                    dx, dy = random.uniform(-W*2.2//5, W*2.2//5), random.uniform(-H*2.2//5, H*2.2//5)
                pos_list.append([dx,dy])
                param = [f,0,W/2-f*W/2-f*dx,0,f,H/2-f*H/2-f*dy,0,0]
                params.append(param)
            params += [[1,0,0,0,1,0,0,0]]*(max_num-number)
            texes += [backgrounds[0]]*(max_num-number)
            masks += [ani_masks[0]]*(max_num-number)

            yield bg, np.stack(texes, axis=0), np.stack(masks, axis=0), params


def generate_image(bg, texes, masks, params, max_num):
    #given the randomly selected foregrounds, backgrounds and their factors of variation, synthesize the final image.

    #zoom and shift transform
    texes = tf.contrib.image.transform(texes, transforms=params,interpolation='BILINEAR')
    masks = tf.contrib.image.transform(masks, transforms=params,interpolation='BILINEAR')
    texes = tf.clip_by_value(texes, 0, 1)
    masks = tf.clip_by_value(masks, 0, 1)

    cum_mask = tf.zeros_like(masks[0])
    depth_masks = [] 
    perturbed_texes = []
    for i in range(0,max_num):  #depth order -> from near to far
        perturbed_tex = texes[i]+tf.random.uniform([], minval=-0.36,maxval=0.36, dtype=tf.float32)  #brightness variation
        perturbed_texes.append(perturbed_tex)
        m = masks[i]*(1-cum_mask)
        cum_mask += m
        depth_masks.append(m)
    

    bg = bg + tf.random.uniform([], minval=-0.36,maxval=0.36, dtype=tf.float32)
    perturbed_texes = [bg] + perturbed_texes
    depth_masks = [1-cum_mask] + depth_masks
    perturbed_texes = tf.stack(perturbed_texes, axis=0) #C H W 3
    depth_masks = tf.stack(depth_masks, axis=0) #C H W 1

    img = tf.reduce_sum(perturbed_texes*depth_masks, axis=0) #H W 3


    data = {}
    data['img'] = img  #float 0~1
    data['masks'] = tf.transpose(depth_masks, perm=[1,2,3,0])  #C H W 1 -> H W 1 C  (bg first channel)
    return data


def dataset(data_path, batch_size, max_num=5, phase='train'):
    """
    Args:
    data_path: the path of npz
    batch_size: batchsize
    max_num: max number of animals in an image
    phase: train: infinitely online generating image/ val: deterministic 200 / test: deterministic 2000
    """
    assert max_num==5,'please re-assign the distribution in get_number(), flying_animals_utils.py, if you need to reset max_num'
    data = np.load(data_path) 
    if phase in ['train', 'test']: 
        assert 200%batch_size==0
    deterministic_params = generate_params(data, max_num=max_num, num=200 if phase=='val' else 2000) if not phase=='train' else None 
    partial_fn = functools.partial(params_gen, data=data, max_num=max_num, 
        deterministic_params=deterministic_params) 
    dataset = tf.data.Dataset.from_generator(
        partial_fn,
        (tf.float32, tf.float32, tf.float32, tf.float32),
        (tf.TensorShape([H,W,3]),tf.TensorShape([max_num,H,W,3]),tf.TensorShape([max_num,H,W,1]),tf.TensorShape([max_num,8])))
    dataset = dataset.map(lambda bg,t,m,p: generate_image(bg,t,m,p,max_num=max_num), 
        num_parallel_calls=1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    return dataset