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
from model.train_graph import Train_Graph
import imageio
import numpy as np
import time


# for evaluate CIS term

def convert(img):
    #float32 0~1 -> uint8 0~255
    return (img*255).astype(np.uint8)

def eval(FLAGS):
    # learner
    graph = Train_Graph(FLAGS)  
    graph.build()

    #restore_vars = tf.trainable_variables('Inpainter')+tf.trainable_variables('Generator')
    restore_vars = tf.global_variables('Inpainter') + tf.global_variables('Generator')
    saver = tf.train.Saver(restore_vars)
    #training =True??
    # var_dict = {}
    # for v in tf.trainable_variables('Generator'):
    #     if 'aspp' in v.op.name:
    #         name = 'resnet_v2_50/'+v.op.name[v.op.name.find('aspp'):]
    #     else:
    #         name = 'resnet_v2_50/'+v.op.name[v.op.name.find('resnet_v2/')+10:]
    #     var_dict[name] = v
    # saver = tf.train.Saver(var_dict)
    
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(graph.val_iterator.initializer)
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
            #'edge_map': graph.edge_map,
            'loss_Generator_______': graph.loss['Generator'], #scalar
            'loss_Generator_branch': graph.loss['Generator_branch'],  #B,C
            'loss_Generator_denomi': graph.loss['Generator_denominator'], #B,C
            'loss_Generator_numera': graph.loss['Generator_numerator'],  #B,C
            'loss_GroundTru_______': graph.loss['GT'],
            'loss_GroundTru_branch': graph.loss['GT_branch'],
            'loss_GroundTru_denomi': graph.loss['GT_denominator'],
            'loss_GroundTru_numera': graph.loss['GT_numerator']
        }
        if FLAGS.dataset == 'flying_animals':
            score = [[]]*FLAGS.max_num
            for i in range(FLAGS.bg_num):  #default output 100 background examples
                results = sess.run(fetches, feed_dict={graph.is_training: False})
                for j in range(FLAGS.max_num):
                    #score[j].append(Permute_IoU(results['GT_masks'][j], results['generated_masks'][j]))
                    s = Permute_IoU(label=results['GT_masks'][j], pred=results['generated_masks'][j])
                    score[j] = score[j] + [s]                           
                    # if i <=9:
                    #     tex = results['image_batch'][j,:,:,:]
                    #     for k in range(FLAGS.num_branch):
                    #         mask = results['generated_masks'][j,:,:,:,k]
                    #         show = (tex*mask*255).astype(np.uint8)
                    #         imageio.imwrite(os.path.join(FLAGS.checkpoint_dir,'{}{}_mask{}.png').format(i,j,k), show)
            score = np.array(score) # max_num*100
            bg_score = np.mean(score, axis=0)#100,
            ob_score = np.mean(score, axis=1)#max_num,

            with open(os.path.join(FLAGS.checkpoint_dir,'evaluation_results.txt'),'w') as f:
                for i in range(FLAGS.max_num):
                    f.write("Object {}: {}\n".format(i+1, ob_score[i]))
                f.write("------------------------------------------------\n")

                #bg_score_sorted = np.sort(bg_score, axis=None)
                bg_ind = np.argsort(bg_score)
                for j in range(FLAGS.bg_num):
                    f.write("background{}: {}\n".format(bg_ind[j]+1, bg_score[bg_ind[j]]))
                f.write("------------------------------------------------\n")
                score = np.reshape(score,[-1])
                score_sorted = np.sort(score, axis=None)
                score_sorted = score_sorted[::-1]
                for i in range(0,FLAGS.max_num*FLAGS.bg_num,10):
                    f.write("{}%: {}\n".format(i*100/(FLAGS.max_num*FLAGS.bg_num), score_sorted[i]))
        elif FLAGS.dataset == 'multi_texture':
            score = [[]]*FLAGS.max_num
            for i in range(50):
                for j in range(FLAGS.max_num):
                    results = sess.run(fetches, feed_dict={graph.is_training: False})
                    s = Permute_IoU(label=results['GT_masks'][j], pred=results['generated_masks'][j])
                    score[j] = score[j]+[s]

                    if j == 0 and i < 4:
                        newdir = os.path.join(FLAGS.checkpoint_dir, str(i))
                        if not os.path.exists(newdir):
                            os.makedirs(newdir)
                        tex = results['image_batch'][j,:,:,:]
                        for k in range(FLAGS.num_branch):
                            mask = results['generated_masks'][j,:,:,:,k]
                            show = (tex*mask*255).astype(np.uint8)
                            imageio.imwrite(os.path.join(newdir,'{}mask.png').format(k), show)
                            pred_tex = results['pred_intensities'][j,:,:,:,k]
                            show = (pred_tex*mask*255).astype(np.uint8)
                            imageio.imwrite(os.path.join(newdir,'{}mask_pred.png').format(k), show)
                            
                        with open(os.path.join(newdir, 'loss.txt'), 'w') as f:
                            loss_Generator = np.sum(results['loss_Generator_branch'][0,:])
                            loss_GT = np.sum(results['loss_GroundTru_branch'][0,:])
                            f.write("loss_GroundTruth {:>7.3f} \n".format(loss_GT))
                            f.write("loss_Generator   {:>7.3f} \n".format(loss_Generator))
                            f.write("\n")
                            for m in range(FLAGS.num_branch):
                                f.write("branch_{} \n".format(m))
                                f.write("loss_Generator_branch {:>7.3f}   ".format(results['loss_Generator_branch'][0,m]))
                                f.write("loss_Generator_denomi {:>7.3f}   ".format(results['loss_Generator_denomi'][0,m]))
                                f.write("loss_Generator_numera {:>7.3f}   \n".format(results['loss_Generator_numera'][0,m]))
                                f.write("loss_GroundTru_branch {:>7.3f}   ".format(results['loss_GroundTru_branch'][0,m]))
                                f.write("loss_GroundTru_denomi {:>7.3f}   ".format(results['loss_GroundTru_denomi'][0,m]))
                                f.write("loss_GroundTru_numera {:>7.3f}   \n".format(results['loss_GroundTru_numera'][0,m]))
                                f.write("\n") 
            with open(os.path.join(FLAGS.checkpoint_dir,'evaluation_results.txt'),'w') as f:
                #mean variance
                for i in range(FLAGS.max_num):
                    f.write("Object {}: mean {} variance{}\n".format(i+1, np.mean(score[i]), np.var(score[i])))

        myprint("save results in "+FLAGS.checkpoint_dir)
 


            
