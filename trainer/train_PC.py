import tensorflow as tf 
import os
import gflags  
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

def train(FLAGS):
    # learner
    graph = Train_Graph(FLAGS)  
    graph.build()

    summary_op, eval_summary_op = Summary.collect_PC_summary(graph, FLAGS)
    saver_vars = [v for v in tf.global_variables('Inpainter')+tf.global_variables('Generator')+ \
            tf.global_variables('VAE') + tf.global_variables('Fusion') if not 'Adam' in v.op.name] 
    saver = tf.train.Saver(saver_vars, max_to_keep=100)
    sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.checkpoint_dir, "end2end_Sum"),
                                 saver=None, save_summaries_secs=0) 

    with sv.managed_session() as sess:
        assert os.path.isfile(FLAGS.fullmodel_ckpt+'.index')
        saver.restore(sess, FLAGS.fullmodel_ckpt)
        myprint ("Finetune model {} for perceptual consistency".format(FLAGS.fullmodel_ckpt))
        myinput('Press enter to continue')

        start_time = time.time()
        step = sess.run(graph.global_step)
        progbar = Progbar(target=FLAGS.ckpt_steps) #100k
        sum_iters = FLAGS.iters_gen + FLAGS.iters_inp

        while (time.time()-start_time)<FLAGS.max_training_hrs*3600:
            if sv.should_stop():
                break
            fetches = {"global_step_inc": graph.incr_global_step, "step": graph.global_step}
            if step%sum_iters < FLAGS.iters_inp:
                fetches['train_op'] = graph.train_ops['Inpainter']
            else:
                fetches['train_op'] = graph.train_ops['Generator']

            if step % FLAGS.summaries_steps == 0:
                fetches['summary'] = summary_op

            results = sess.run(fetches, feed_dict={graph.is_training: True})
            progbar.update(step%FLAGS.ckpt_steps)
            if step % FLAGS.summaries_steps == 0:
                print ("   Step:%3dk time:%4.4fmin" \
                    %(step/1000, (time.time()-start_time)/60))
                sv.summary_writer.add_summary(results['summary'], step)
            

            if step % FLAGS.ckpt_steps == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=step)
                progbar = Progbar(target=FLAGS.ckpt_steps)

            if step % (100*FLAGS.summaries_steps) == 0 and not step==0:
                #evaluation
                sess.run(graph.val_iterator.initializer)
                fetches = {'GT_masks':graph.GT_masks, 'generated_masks':graph.generated_masks}
                num_sample = 9*9*9*9-1
                niter = num_sample//FLAGS.batch_size
                assert num_sample%FLAGS.batch_size==0
                score = 0
                arg_maxIoUs = []
                for it in range(niter):
                    results_val = sess.run(fetches, feed_dict={graph.is_training:False})
                    for k in range(FLAGS.batch_size):
                        k_score, arg_maxIoU= Permute_IoU(label=results_val['GT_masks'][k], pred=results_val['generated_masks'][k])
                        score += k_score
                        arg_maxIoUs.append(arg_maxIoU)

                score = score/num_sample
                arg_maxIoUs = np.stack(arg_maxIoUs, axis=0) #400, 3
                count = np.sum(arg_maxIoUs, axis=0) #3    0 square // 1 ellipse // 2 background
                switching_rate = np.min(count)/num_sample
                eval_summary = sess.run(eval_summary_op, feed_dict={graph.loss['EvalIoU_var']: score, 
                    graph.switching_rate: switching_rate})
                sv.summary_writer.add_summary(eval_summary, step)

            step = results['step']

        myprint("Training completed")