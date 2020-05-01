import tensorflow as tf
import numpy as np
import random
import imageio
import os
import functools
import scipy.ndimage as ndimage
import math


# data generation script for data generation
# var_range  the color variation
# number of thread
# num_parallel_calls=1 to output deterministic result

#if any modification is done, please check git status and git diff
H, W = 192,256
def convert(imgs):
    return (imgs/255).astype(np.float32)

rm_class = [2,3,7,8,11,20]
rm_class = [2,3,7,8,11,20,4,6,9,19]  #24-10=14 category
fg_select = [i for i in range(0,241) if not math.ceil(i/10)-1 in rm_class]

rm_bg = [34,80,71,72,38,28,85,88,93,86,75,70,42,30,24,64,65,90,82,25]  
#remain 80bg  (0.7639+-0.006)  （0.7821+-0.005）
rm_bg = [34,80,71,72,38,28,85,88,93,86,75,70,42,30,24,64,65,90,82,25,29,21,40,4,51,79,33,68,83,91,59,6,87,45,94,99] 
#remain 65 bg  (0.7735+-0.004)   (0.7988+-0.002)
rm_bg = [34,80,71,72,38,28,85,88,93,86,75,70,42,30,24,64,65,90,82,25,
29,21,40,4,51,79,33,68,83,91,59,6,87,45,94,99,
23,78,36,19,77,39,62,52,81,56,98,43] 
#remain 53 bg (0.7843+-0.005)

bg_select = [i for i in range(0,101) if not i in rm_bg]
#print(len(bg_select))
#81
def check_occlusion(pos, pos_list):
    #to prevent severe occlusion
    #pos[x,y] position to add
    #pos_list [[],[],] previous position
    #return True if pos[x,y] largely overlaps with one of previous positions
    for p in pos_list:
        dist = abs(pos[0]-p[0])+abs(pos[1]-p[1])
        if dist<=97:
            return True
    return False

def generate_params(max_num, data):
    deter_params = []
    random.seed(1365)
    for i in range(0,10): #[1, len(bg_select)-1]
        for j in range(0,200):#range(1,max_num+1):   #1，2，3，4，max_num
            param = dict()
            param['bg_index'] = bg_select[random.randint(1, len(bg_select)-1)]
            param['number'] = get_number(j/200)#j 
            param['fg_indices'] = [fg_select[random.randint(1,len(fg_select)-1)] for k in range(param['number'])] +[0]*(max_num-param['number']) #pad
            pos_list = []
            params = []
            for k in range(param['number']): 
                f = random.uniform(0.9, 1.2)   #input /output
                #rotate = random.gauss(0, 4) if rotation else 0
                dx, dy = random.uniform(-W*2.2//5, W*2.2//5), random.uniform(-H*2.2//5, H*2.2//5)
                while check_occlusion([dx, dy], pos_list):
                    dx, dy = random.uniform(-W*2.2//5, W*2.2//5), random.uniform(-H*2.2//5, H*2.2//5)
                pos_list.append([dx,dy])
                p= [f,0,W/2-f*W/2-f*dx,0,f,H/2-f*H/2-f*dy,0,0]

                params.append(p)
            params += [[1,0,0,0,1,0,0,0]]*(max_num-param['number'])
            param['params'] = params
            deter_params.append(param)
    return deter_params  # list of dicts 5*100 element

def get_number(n):
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


    step = 0
    while True:
        step += 1
        param = deterministic_params[(step-1)%(200*10)]
        bg = backgrounds[param['bg_index']]
        texes = [foregrounds[i] for i in param['fg_indices']]
        masks = [ani_masks[i] for i in param['fg_indices']]
        fg_indices = param['fg_indices']
        yield bg, np.stack(texes, axis=0), np.stack(masks, axis=0), param['params'], fg_indices



def random_compose(bg, texes, masks, params, fg_indices, max_num, var_range):

    #zoom and shift transform
    masks0 = masks
    texes0 = texes
    texes = tf.contrib.image.transform(texes, transforms=params,interpolation='BILINEAR')
    masks = tf.contrib.image.transform(masks, transforms=params,interpolation='BILINEAR')
    texes = tf.clip_by_value(texes, 0, 1)
    masks = tf.clip_by_value(masks, 0, 1)
    #masks = 1-tf.nn.max_pool2d(1-masks, ksize=3, strides=1, padding='SAME',data_format='NHWC')
    #combine
    cum_mask = tf.zeros_like(masks[0])
    depth_masks = []
    hue_texes = []
    for i in range(0,max_num):
        hue_tex = texes[i]+tf.random.uniform([], minval=-1.0*var_range,maxval=var_range, dtype=tf.float32)
        hue_texes.append(hue_tex)
        #m = 1 - tf.nn.max_pool2d(1-masks[i], ksize=3, strides=1, padding='SAME')
        m = masks[i]*(1-cum_mask)
        cum_mask += m
        depth_masks.append(m)
    

    bg = bg + tf.random.uniform([], minval=-1.0*var_range,maxval=var_range, dtype=tf.float32)
    hue_texes = [bg] + hue_texes
    depth_masks = [1-cum_mask] + depth_masks
    hue_texes = tf.stack(hue_texes, axis=0) #C H W 3
    depth_masks = tf.stack(depth_masks, axis=0) #C H W 1

    img = tf.reduce_sum(hue_texes*depth_masks, axis=0) #H W 3


    data = {}
    data['img'] = img
    #data['texes'] = tf.transpose(hue_texes, perm=[1,2,3,0])
    data['masks'] = tf.transpose(depth_masks, perm=[1,2,3,0])  #C H W 1 -> H W 1 C  (bg first channel)
    data['fg_indices'] = fg_indices #C,
    data['masks0'] = tf.transpose(masks0, perm=[1,2,3,0])
    data['texes0'] = tf.transpose(texes0, perm=[1,2,3,0])
    return data


def dataset(data_path, batch_size, max_num=5):
    """
    Args:
    data_path: the path of combination elements  
    (min_num, max_num): number of the animals in an image
    val: validation dataset when True  #produce deterministic results each time dataloader is initialized
    """
    assert max_num==5 
    var_range = 0 
    data = np.load(data_path) 
    deterministic_params = generate_params(max_num,data) 
    partial_fn = functools.partial(params_gen, data=data, max_num=max_num, 
        deterministic_params=deterministic_params) 
    dataset = tf.data.Dataset.from_generator(
        partial_fn,
        (tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        (tf.TensorShape([H,W,3]),tf.TensorShape([max_num,H,W,3]),tf.TensorShape([max_num,H,W,1]),tf.TensorShape([max_num,8]),
            tf.TensorShape([max_num])))
    dataset = dataset.map(lambda bg,t,m,p,f: random_compose(bg,t,m,p,f,max_num=max_num, var_range=var_range), 
        num_parallel_calls=1)

    dataset = dataset.batch(batch_size)
    # # print (dataset)
    dataset = dataset.prefetch(10)
    return dataset


# random.seed(4)
# sess = tf.Session()
# bs = 5
# ds = dataset('../../data/flying_animals_data/img_data.npz', bs,  max_num=5, val=True)
# iterator = ds.make_initializable_iterator()
# getnext = iterator.get_next()
# sess.run(iterator.initializer)
# for i in range(len(bg_select)-1):
#     niter = 100//bs
#     for it in range(niter):
#         data = sess.run(getnext)
#         for j in range(bs):
#             if j == 0 and it==0:
#                 img = (data['img'][j]*255).astype(np.uint8)
#                 imageio.imwrite('debug/{}{}.png'.format(i,j), img)
#                 fg_indices = data['fg_indices']
#                 print('fg_indices', fg_indices)

# sess.run(iterator.initializer)
# for i in range(2):
#     data = sess.run(getnext)
#     for j in range(bs):
#         img = (data['img'][j]*255).astype(np.uint8)
#         imageio.imwrite('debug2/{}{}.png'.format(i,j), img)