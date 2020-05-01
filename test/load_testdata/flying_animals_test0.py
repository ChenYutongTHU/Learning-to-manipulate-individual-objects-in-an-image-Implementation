import tensorflow as tf
import numpy as np
import os
import functools
import imageio
def generate_path(data_path, number, num_branch=6):
    for i in range(number):
        img_path = os.path.join(data_path, 'images', '{:0>4d}.png'.format(i))
        mask_path = [os.path.join(data_path, 'masks', '{:0>4d}_mask{}.png'.format(i,j)) for j in range(num_branch)]
        yield img_path, mask_path

def read(img_path, masks_paths, num_branch):
    image = tf.io.read_file(img_path)
    img = tf.io.decode_png(image, channels=3)
    img = tf.cast(img/255, tf.float32)

    masks =[]
    for i in range(num_branch):
        mask = tf.io.read_file(masks_paths[i])
        mask = tf.io.decode_png(mask, channels=1)
        mask = tf.cast(mask/255, tf.float32)
        masks.append(mask)
    masks = tf.stack(masks, axis=-1) #H W 1 C

    data={'img':img, 'masks':masks}
    return data
def dataset(data_path, batch_size, number, num_branch=6):
    #data_path/images/0000.png
    #data_path/masks/0000_mask0.png
    generate_fun = functools.partial(generate_path, data_path=data_path, number=number, num_branch=num_branch)
    dataset = tf.data.Dataset.from_generator(generate_fun,(tf.string,tf.string))
    dataset = dataset.map(lambda img_path, masks_paths: read(img_path,masks_paths,num_branch), num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset

# path = '/home/yutong/Learning-to-Manipulate-Individual-Objects-in-an-Image/outputs/fa/CIS_prob/70bf8_bg4_eps100_12_1e-4_1e-4/110k/0.81/DB70bf8_5object_includemask/'
# ds = dataset(data_path=path,batch_size=3,number=100,num_branch=6)
# sess = tf.Session()
# iterator = ds.make_initializable_iterator()
# getnext = iterator.get_next()
# sess.run(iterator.initializer)
# for i in range(2):
#     data = sess.run(getnext)
#     for j in range(3):
#         img = (data['img'][j,:,:,:]*255).astype(np.uint8)
#         imageio.imwrite(os.path.join('debug','{}img.png'.format(j)), img)
#         mk = (data['masks'][j,:,:,:,0]*255).astype(np.uint8)
#         imageio.imwrite(os.path.join('debug','{}mask.png'.format(j)), mk)