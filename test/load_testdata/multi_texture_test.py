import tensorflow as tf
import numpy as np
import random
import imageio
import os
import functools
import scipy.ndimage as ndimage
H,W=64,64


def check_occlusion(pos, pos_list):
    #to prevent severe occlusion
    #pos[x,y] position to add
    #pos_list [[],[],] previous position
    #return True if pos[x,y] largely overlaps with one of previous positions
    for p in pos_list:
        dist = abs(pos[0]-p[0])+abs(pos[1]-p[1])
        if dist<=10:
            return True
    return False

def get_number(n):
    if n<=0.8:
        return 4  #0.8
    elif n<=0.9:
        return 3  #0.1
    elif n<=0.97:
        return 2  #0.07
    else:
        return 1 #0.03
def generate_params(data_path, max_num=4, zoom=False, rotation=False, texture_transform=False):
    deterministic_params = []
    random.seed(1365)
    for i in range(10): #10 subset
        for j in range(200):  #200 images for each subset
            param = dict()
            param['number'] = get_number((j%100+1)/100)#j
            param['ind'], param['angle'], param['mat'] = [], [],[]
            pos_list = []
            for k in range(param['number']):
                ind = random.randint(0,1)
                param['ind'].append(ind)
                param['angle'].append(random.uniform(0, 360) if rotation else 0)

                f = random.uniform(0.5,1.2) if zoom else 1  #zoom the shape
                dr = random.uniform(-H/2,H/2)
                dc = random.uniform(-W/2,W/2)
                while check_occlusion([dr,dc], pos_list):
                    dr = random.uniform(-H/2,H/2)
                    dc = random.uniform(-W/2,W/2)
                pos_list.append([dr,dc])
                mat = np.zeros([3,4])
                mat[0][0], mat[0][3] = 1/f, H-H/f+dr
                mat[1][1], mat[1][3] = 1/f, W-W/f+dc
                mat[2][2] = 1
                param['mat'].append(mat)
            param['hue'] = [random.uniform(-0.45,0.45) for h_ in range(max_num+1)]
            deterministic_params.append(param)
    return deterministic_params
    
def multi_texture_gen(data_path, max_num=4, zoom=False, rotation=False, texture_transform=False, deterministic_params=None):
#./ data_path should include bg.png tex.png shape.png  the elements to randomly select
#yield  (number of shapes (int), masks (float32 0~1)[count,H,W,1], textures(float32 0~255)[count,H,W,1])
    tex0  = imageio.imread(os.path.join(data_path,'tex.png'))
    tex0 = (tex0/255).astype(np.float32)
    #heart = imageio.imread(os.path.join(data_path,'heart_2.png')).reshape(64,64,1)
    #heart = (heart/255).astype(np.float32)
    square = imageio.imread(os.path.join(data_path,'square_2.png')).reshape(64,64,1)
    square = (square/255).astype(np.float32)
    ellipse = imageio.imread(os.path.join(data_path,'ellipse_2.png')).reshape(64,64,1)
    ellipse = (ellipse/255).astype(np.float32)
    masks = [ellipse, square]#, heart]

    step = 0
    while True:
        param = deterministic_params[step%(10*200)] if deterministic_params else None
        step += 1
        number = param['number'] if param else get_number(random.uniform(0,1))
        shape_masks = []
        shape_texes = []  #
        cum_mask = np.zeros_like(masks[0])  # occlusion
        pos_list = []
        for i in range(number):  #place the randomly selected and transformed shape on the background in depth order  
            ind = param['ind'][i] if param else random.randint(0, 1)  #choose the shape 
            shape = masks[ind].copy()
            tex = tex0.copy()
            #rotation
            if rotation:
                angle = param['angle'][i] if param else random.uniform(0, 360) 
                shape = ndimage.rotate(shape, angle, reshape=False)
                if texture_transform:
                    tex = ndimage.rotate(tex, angle, reshape=False)
            #zoom and shift
            if param:
                mat = param['mat'][i]
            else:
                f = random.uniform(0.5,1.2) if zoom else 1  #zoom the shape
                dr = random.uniform(-H/2,H/2)
                dc = random.uniform(-W/2,W/2)
                while check_occlusion([dr,dc], pos_list):
                    dr = random.uniform(-H/2,H/2)
                    dc = random.uniform(-W/2,W/2)
                pos_list.append([dr,dc])
                mat = np.zeros([3,4])
                mat[0][0], mat[0][3] = 1/f, H-H/f+dr
                mat[1][1], mat[1][3] = 1/f, W-W/f+dc
                mat[2][2] = 1
            shape = ndimage.affine_transform(shape, mat, output_shape=(64,64,1))
            if texture_transform:
                tex = ndimage.affine_transform(tex, mat, output_shape=(64,64,3))
            #affine_transformed shape might be out of range (0,1)
            shape = np.clip(shape, 0,1)
            shape = shape*(1-cum_mask)  #occluded by other shapes
            #clip the tex to (0~1)
            tex = np.clip(tex, 0, 1)
            cum_mask += shape
            shape_masks.append(shape)
            shape_texes.append(tex)
        for i in range(number, max_num):#pad the returned element
            shape_masks.append(np.zeros_like(shape))
            shape_texes.append(np.zeros_like(tex))
        hue_value = param['hue'] if deterministic_params else [random.uniform(-0.45,0.45) for h_ in range(max_num+1)]
        yield (number, np.stack(shape_masks, axis=0), np.stack(shape_texes, axis=0), hue_value)

def combine(number, masks, texes, hue_value, bg,  max_num):
    #masks C H W 1
    #texes C H W 3
    #bg H W 1
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


def dataset(data_path, batch_size, max_num=4, zoom=False, rotation=False, texture_transform=False):
    """
    Args:
    data_path: the path of combination elements  
    """
    assert max_num==4
    deterministic_params = generate_params(data_path,
        max_num=max_num,zoom=zoom,
        rotation=rotation,texture_transform=texture_transform) 

    partial_fn = functools.partial(multi_texture_gen, 
      data_path=data_path, max_num=max_num, zoom=zoom, rotation=rotation, texture_transform=texture_transform,
      deterministic_params=deterministic_params)

    dataset = tf.data.Dataset.from_generator(
        partial_fn,#(data_path, max_num, zoom, rotation),
        (tf.int32, tf.float32, tf.float32, tf.float32),
        (tf.TensorShape([]),tf.TensorShape([max_num,H,W,1]), tf.TensorShape([max_num,H,W,3]), tf.TensorShape([max_num+1])))

    bg0 = imageio.imread(os.path.join(data_path,'bg.png'))
    bg0 = tf.convert_to_tensor(bg0/255, dtype=tf.float32)  #0~1
    num_parallel_calls = 1 
    dataset = dataset.map(lambda n,m,t,h: combine(n,m,t,h, bg=bg0,max_num=max_num), num_parallel_calls=1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    return dataset
