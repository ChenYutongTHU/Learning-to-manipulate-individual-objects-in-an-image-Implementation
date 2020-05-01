import tensorflow as tf 
import os
import gflags  
#https://github.com/google/python-gflags
import sys
sys.path.append("..") 
import pprint
from keras.utils.generic_utils import Progbar
import model.Summary as Summary
from model.utils.generic_utils import myprint, myinput
from model.eval_graph import Eval_Graph
import imageio
import numpy as np
import time


# for evaluate CIS term

def convert(img):
    #float32 0~1 -> uint8 0~255
    return (img*255).astype(np.uint8)

def eval(FLAGS):
    # learner
    graph = Eval_Graph(FLAGS)  
    graph.build()

    restore_vars = tf.trainable_variables('Inpainter')+tf.trainable_variables('Generator')
    saver = tf.train.Saver(restore_vars)
    #training =True??
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        assert os.path.isfile(FLAGS.fullmodel_ckpt+'.index')
        saver.restore(sess, FLAGS.fullmodel_ckpt)
        myprint("resume model {}".format(FLAGS.fullmodel_ckpt))
        fetches = {
            'image_batch': graph.image_batch, 
            'generated_masks': graph.generated_masks, 
            'GT_masks': graph.GT_masks,
            'pred_intensities': graph.pred_intensities,
            'pred_intensities_GT': graph.pred_intensities_GT,
            'unconditioned_mean': graph.unconditioned_mean,
            'edge_map': graph.edge_map,
            'loss_Generator_______': graph.loss['Generator'],
            'loss_Generator_branch': graph.loss['Generator_branch'],
            'loss_Generator_denomi': graph.loss['Generator_denominator'],
            'loss_Generator_numera': graph.loss['Generator_numerator'],
            'loss_GroundTru_______': graph.loss['GT'],
            'loss_GroundTru_branch': graph.loss['GT_branch'],
            'loss_GroundTru_denomi': graph.loss['GT_denominator'],
            'loss_GroundTru_numera': graph.loss['GT_numerator']
        }
        for i in range(32):  #default output 4 examples
            results = sess.run(fetches)
            dirname = os.path.join(FLAGS.checkpoint_dir, str(i))
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            img = convert(results['image_batch'][0])
            imageio.imwrite(os.path.join(dirname, 'a_img.png'),img)

            for m in range(FLAGS.max_num+1):
                mask = convert(results['generated_masks'][0,:,:,:,m]*results['image_batch'][0]+\
                    (1-results['generated_masks'][0,:,:,:,m])*0.2*results['image_batch'][0])
                imageio.imwrite(os.path.join(dirname,'generated_masks{}.png'.format(m)), mask)

                pred = convert(results['pred_intensities'][0,:,:,:,m]*results['generated_masks'][0,:,:,:,m])
                imageio.imwrite(os.path.join(dirname,'pred_intensities{}.png'.format(m)), pred)

                GTpred = convert(results['pred_intensities_GT'][0,:,:,:,m]*results['GT_masks'][0,:,:,:,m])
                imageio.imwrite(os.path.join(dirname,'pred_intensities_GT{}.png'.format(m)), GTpred)


            mean = convert(results['unconditioned_mean'][0])
            imageio.imwrite(os.path.join(dirname,'unconditioned_mean.png'), mean)

            edge = np.concatenate([results['edge_map'][0],np.zeros_like(results['edge_map'][0,:,:,0:1])], axis=-1)#H W 2+1
            edge = convert(edge)
            imageio.imwrite(os.path.join(dirname,'edge_map.png'), edge)

            with open(os.path.join(dirname, 'loss.txt'), 'w') as f:
                loss_Generator = np.sum(results['loss_Generator_branch'][0,:])
                loss_GT = np.sum(results['loss_GroundTru_branch'][0,:])
                f.write("loss_GroundTruth {:>7.3f} \n".format(loss_GT))
                f.write("loss_Generator   {:>7.3f} \n".format(loss_Generator))
                f.write("\n")
                for m in range(FLAGS.max_num+1):
                    f.write("branch_{} \n".format(m))
                    f.write("loss_Generator_branch {:>7.3f}   ".format(results['loss_Generator_branch'][0,m]))
                    f.write("loss_Generator_denomi {:>7.3f}   ".format(results['loss_Generator_denomi'][0,m]))
                    f.write("loss_Generator_numera {:>7.3f}   \n".format(results['loss_Generator_numera'][0,m]))
                    f.write("loss_GroundTru_branch {:>7.3f}   ".format(results['loss_GroundTru_branch'][0,m]))
                    f.write("loss_GroundTru_denomi {:>7.3f}   ".format(results['loss_GroundTru_denomi'][0,m]))
                    f.write("loss_GroundTru_numera {:>7.3f}   \n".format(results['loss_GroundTru_numera'][0,m]))
                    f.write("\n")
        myprint("save results in "+dirname)
 


            
