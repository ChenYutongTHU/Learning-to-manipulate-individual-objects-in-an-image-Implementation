import tensorflow as tf
import numpy as np
import random
import imageio
import os
import functools
import scipy.ndimage as ndimage
H,W=64,64
def check_occlusion(pos, pos_list):
    for p in pos_list:
        dist = abs(pos[0]-p[0])+abs(pos[1]-p[1])
        if dist<=10:
            return True
    return False

def generate_params(data_path, num, max_num):
    deterministic_params = []
    for i in range(num):
        param = dict()
        param['number'] = get_number(random.uniform(0,1))
        param['ind'], param['mat'] = [],[]
        pos_list = []
        for k in range(param['number']):
            param['ind'].append(random.randint(0,1))  #shape 0 square 1 ellipse
            dr = random.uniform(-H/2,H/2)
            dc = random.uniform(-W/2,W/2)
            while check_occlusion([dr,dc], pos_list):
                dr = random.uniform(-H/2,H/2)
                dc = random.uniform(-W/2,W/2)
            pos_list.append([dr,dc])
            mat = np.zeros([3,4])
            mat[0][0], mat[0][3] = 1, dr
            mat[1][1], mat[1][3] = 1, dc
            mat[2][2] = 1
            param['mat'].append(mat)
        param['hue'] = [random.uniform(-0.45,0.45) for h_ in range(max_num+1)]
        deterministic_params.append(param)
    return deterministic_params

def multi_texture_gen(data_path, max_num=4, deterministic_params=None):
    tex0  = imageio.imread(os.path.join(data_path,'tex.png'))
    tex0 = (tex0/255).astype(np.float32)
    square = imageio.imread(os.path.join(data_path,'square_2.png')).reshape(64,64,1)
    square = (square/255).astype(np.float32)
    ellipse = imageio.imread(os.path.join(data_path,'ellipse_2.png')).reshape(64,64,1)
    ellipse = (ellipse/255).astype(np.float32)
    masks = [ellipse, square]

    step = 0
    while True:
        param = deterministic_params[step%(len(deterministic_params))] if deterministic_params else None
        step += 1
        number = param['number'] if param else get_number(random.uniform(0,1))
        shape_masks = []
        shape_texes = []  #
        cum_mask = np.zeros_like(masks[0])  # occlusion
        pos_list = []
        for i in range(number):  #place the randomly selected and transformed shape on the background in the ascending depth order 
            ind = param['ind'][i] if param else random.randint(0, 1)  #choose the shape 
            shape = masks[ind].copy()
            tex = tex0.copy()
            if param:
                mat = param['mat'][i]
            else:
                dr = random.uniform(-H/2,H/2)
                dc = random.uniform(-W/2,W/2)
                while check_occlusion([dr,dc], pos_list):
                    dr = random.uniform(-H/2,H/2)
                    dc = random.uniform(-W/2,W/2)
                pos_list.append([dr,dc])
                mat = np.zeros([3,4])
                mat[0][0], mat[0][3] = 1, dr
                mat[1][1], mat[1][3] = 1, dc
                mat[2][2] = 1
            shape = ndimage.affine_transform(shape, mat, output_shape=(64,64,1))
            shape = np.clip(shape, 0,1)
            shape = shape*(1-cum_mask)  
            cum_mask += shape
            shape_masks.append(shape)
            shape_texes.append(tex)
        for i in range(number, max_num):#pad the returned element
            shape_masks.append(np.zeros_like(shape))
            shape_texes.append(np.zeros_like(tex))
        hue_value = param['hue'] if deterministic_params else [random.uniform(-0.45,0.45) for h_ in range(max_num+1)]
        yield (number, np.stack(shape_masks, axis=0), np.stack(shape_texes, axis=0), hue_value)

def get_number(n):
    if n<=0.8:
        return 4  #0.8
    elif n<=0.9:
        return 3  #0.1
    elif n<=0.97:
        return 2  #0.07
    else:
        return 1 #0.03

def combine(number, masks, texes, hue_value, bg,  max_num):
    hue_texes = []
    for i in range(max_num):
      hue_texes.append(tf.image.adjust_hue(texes[i], hue_value[i+1])) #randomly perturb the hue value
    hue_texes = tf.stack(hue_texes, axis=0) #C H W 1
    fg = tf.reduce_sum(masks*hue_texes, axis=0)#np.sum(masks*texes, axis=0, keepdims=False)   #H W 3
    bg_mask = 1-tf.reduce_sum(masks, axis=0) # H W 1
    bg = tf.image.adjust_hue(bg, hue_value[0])

    all_masks = tf.concat([tf.expand_dims(bg_mask,axis=0), masks], axis=0)  #H W 1  + C H W 1

    data = {}
    data['img'] = bg_mask*bg+fg
    data['masks'] = tf.transpose(all_masks, perm=[1,2,3,0])  #C H W 1 -> H W 1 C
    return data

def dataset(data_path, batch_size, max_num=4, phase='train'):
    """
    Args:
    data_path: the path of combination elements  
    batch_size:
    max_num: max_num: max number of objects in an image
    phase: train: infinitely online generating image/ val: deterministic 200 / test: deterministic 2000
    """
    assert max_num==4,'please re-assign the distribution in get_number(), multi_texture_utils.py, if you need to reset max_num'
    
    deterministic_params = generate_params(data_path, num=200 if phase=='val' else 2000, max_num=max_num) if not phase=='train' else None 

    partial_fn = functools.partial(multi_texture_gen, 
      data_path=data_path, max_num=max_num, deterministic_params=deterministic_params)

    dataset = tf.data.Dataset.from_generator(
        partial_fn,#(data_path, max_num, zoom, rotation),
        (tf.int32, tf.float32, tf.float32, tf.float32),
        (tf.TensorShape([]),tf.TensorShape([max_num,H,W,1]), tf.TensorShape([max_num,H,W,3]), tf.TensorShape([max_num+1])))

    bg0 = imageio.imread(os.path.join(data_path,'bg.png'))
    bg0 = tf.convert_to_tensor(bg0/255, dtype=tf.float32)  #0~1
    dataset = dataset.map(lambda n,m,t,h: combine(n,m,t,h, bg=bg0,max_num=max_num), num_parallel_calls=tf.data.experimental.AUTOTUNE if phase=='train' else 1)
    dataset = dataset.batch(batch_size)
    # # print (dataset)
    dataset = dataset.prefetch(10)
    return dataset
