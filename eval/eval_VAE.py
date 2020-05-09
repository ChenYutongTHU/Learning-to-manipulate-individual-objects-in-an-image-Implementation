import tensorflow as tf 
import os
import gflags  
#https://github.com/google/python-gflags
import sys
sys.path.append("..") 
import pprint
from keras.utils.generic_utils import Progbar
import model.Summary as Summary
from model.utils.generic_utils import myprint, myinput, Permute_IoU
from model.traverse_graph import Traverse_Graph
import imageio
import numpy as np
import time


def convert2float(img):
    return (img/255).astype(np.float32)
def convert2int(img):
    return (img*255).astype(np.uint8)


def pad_img(img):
    H,W,C = img.shape
    pad_img = np.ones([H+4, W+4,C])
    pad_img[2:2+H,2:2+W,:] = img  #H W 3
    return pad_img

def eval(FLAGS):
    graph = Traverse_Graph(FLAGS)
    graph.build()

    restore_vars = tf.global_variables('VAE') + tf.global_variables('Generator') + tf.global_variables('Fusion')
    saver = tf.train.Saver(restore_vars)

    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        assert os.path.isfile(FLAGS.fullmodel_ckpt+'.index')
        saver.restore(sess, FLAGS.fullmodel_ckpt)
        myprint("resume model {}".format(FLAGS.fullmodel_ckpt))
        fetches = {
            'image_batch': graph.image_batch, 
            'generated_masks': graph.generated_masks, 
            'traverse_results': graph.traverse_results,
            'out_bg': graph.out_bg,
            'in_bg': graph.in_bg
        }
        assert FLAGS.batch_size==1
        input_img = convert2float(imageio.imread(FLAGS.input_img))
        input_img = np.expand_dims(input_img, axis=0)

        results = sess.run(fetches, feed_dict={graph.image_batch: input_img})
        img = convert2int(results['image_batch'][0])

        imageio.imwrite(os.path.join(FLAGS.checkpoint_dir, 'img.png'), img)
        imageio.imwrite(os.path.join(FLAGS.checkpoint_dir, 'outbg.png'), convert2int(results['out_bg'][0]))
        imageio.imwrite(os.path.join(FLAGS.checkpoint_dir, 'inbg.png'), convert2int(results['in_bg'][0]))
        for i in range(FLAGS.num_branch):
            imageio.imwrite(os.path.join(FLAGS.checkpoint_dir, 'segment_{}.png'.format(i)), convert2int(results['generated_masks'][0,:,:,:,i]*results['image_batch'][0]))

        outputs = np.array(results['traverse_results'])

        if FLAGS.traverse_type=='tex':
            nch = 3
            ndim = FLAGS.tex_dim
        elif FLAGS.traverse_type=='bg':
            nch = 3
            ndim = FLAGS.bg_dim
        else:
            nch = 1
            ndim = FLAGS.mask_dim            

        traverse_branch = [i for i in range(0,FLAGS.num_branch) if FLAGS.traverse_branch=='all' or str(i) in FLAGS.traverse_branch.split(',')]
        traverse_dim = [i for i in range(ndim) if FLAGS.traverse_dim=='all' or str(i) in FLAGS.traverse_dim.split(',')]
        traverse_value = list(np.linspace(-1*FLAGS.traverse_range, FLAGS.traverse_range, 10))


        outputs = np.reshape(outputs, [len(traverse_branch), len(traverse_dim), len(traverse_value),FLAGS.img_height,FLAGS.img_width,-1])
        #tbranch * tdim * step *  H * W * 3

        #show gif
        for i in range(len(traverse_branch)):
            b = traverse_branch[i]
            out = outputs[i] #tdim * step* H * W * 3

            gif_imgs = []
            for j in range(len(traverse_value)):
                value = traverse_value[j]

                group = []
                for d in range(len(traverse_dim)):
                    dim = traverse_dim[d]
                    group.append(pad_img(out[d,j,:,:,:])) #H W 3
                group = np.concatenate(group, axis=1) #H k*W 3
                group = convert2int(group)

                gif_imgs.append(group)

            imageio.mimsave(os.path.join(FLAGS.checkpoint_dir, 'branch{}.gif'.format(b)), gif_imgs)





        
