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
    pad_img = (np.ones([H+4, W+4,C])*255).astype(np.uint8)
    pad_img[2:2+H,2:2+W,:] = img  #H W 3
    return pad_img

def eval(FLAGS):
    graph = Traverse_Graph(FLAGS)
    graph.build()

    restore_vars = tf.global_variables('VAE') + tf.global_variables('Generator') + tf.global_variables('Fusion')
    saver = tf.train.Saver(restore_vars)

    #CIS_saver = tf.train.Saver(tf.global_variables('Generator'))
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        assert os.path.isfile(FLAGS.fullmodel_ckpt+'.index')
        saver.restore(sess, FLAGS.fullmodel_ckpt)
        # CIS_saver.restore(sess, FLAGS.CIS_ckpt)

        #saver.save(sess, '/home/yutong/Learning-to-manipulate-individual-objects-in-an-image-Implementation/save_checkpoint/md/model', global_step=0)
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

        results = sess.run(fetches, feed_dict={graph.image_batch0: input_img})
        img = convert2int(results['image_batch'][0])

        imageio.imwrite(os.path.join(FLAGS.checkpoint_dir, 'img.png'), img)
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

        if FLAGS.traverse_type=='bg':
            traverse_branch = [FLAGS.num_branch-1]
        else:
            traverse_branch = [i for i in range(0,FLAGS.num_branch) if FLAGS.traverse_branch=='all' or str(i) in FLAGS.traverse_branch.split(',')]
        traverse_value = list(np.linspace(FLAGS.traverse_start, FLAGS.traverse_end, 60))


        if FLAGS.dataset == 'flying_animals':
            outputs = np.reshape(outputs, [len(traverse_branch), FLAGS.top_kdim, len(traverse_value),FLAGS.img_height//2,FLAGS.img_width//2,-1])
        else:
            outputs = np.reshape(outputs, [len(traverse_branch), FLAGS.top_kdim, len(traverse_value),FLAGS.img_height,FLAGS.img_width,-1])
        #tbranch * tdim * step *  H * W * 3


        branches = []
        for i in range(len(traverse_branch)):
            values = [[None for jj in range(FLAGS.top_kdim) ] for ii in range(len(traverse_value))]
            b = traverse_branch[i]
            out = outputs[i] #tdim * step* H * W * 3
            for d in range(FLAGS.top_kdim):
                gif_imgs = []
                for j in range(len(traverse_value)):
                    img = (out[d,j,:,:,:]*255).astype(np.uint8)
                    gif_imgs.append(img)
                    values[j][d] = pad_img(img)
                name = 'branch{}_var{}.gif'.format(b, d)
                imageio.mimsave(os.path.join(FLAGS.checkpoint_dir, name), gif_imgs, duration=1/30)

            #values  len(traverse_value) * kdim (img)
            value_slices = [np.concatenate(values[j], axis=1) for j in range(len(traverse_value))]  #group different dimensions along the axis x
            #len(traverse_value)*(H*W*3)
            branches.append(value_slices)  
        merge_slices = [np.concatenate([branches[i][j] for i in range(len(traverse_branch))], axis=0) for j in range(len(traverse_value))]  


        #imageio.mimsave(os.path.join(FLAGS.checkpoint_dir, 'output.gif'), merge_slices, duration=1/30)


        
