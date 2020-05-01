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
import re

def train(FLAGS):
    graph = Train_Graph(FLAGS)  
    graph.build()

    summary_op, generator_summary_op, branch_summary_op, eval_summary_op = Summary.collect_CIS_summary(graph, FLAGS)
    with tf.name_scope("parameter_count"):
        total_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                for v in tf.trainable_variables()])

    save_vars = tf.global_variables('Inpainter')+tf.global_variables('Generator')+ \
        tf.global_variables('train_op') #including global step
    if FLAGS.resume_inpainter:
        assert os.path.isfile(FLAGS.inpainter_ckpt+'.index')
        inpainter_saver = tf.train.Saver(tf.trainable_variables('Inpainter'))#only restore the trainable variables

    if FLAGS.resume_resnet:
        assert os.path.isfile(FLAGS.resnet_ckpt)
        resnet_reader=tf.compat.v1.train.NewCheckpointReader(FLAGS.resnet_ckpt)
        resnet_map = resnet_reader.get_variable_to_shape_map()
        resnet_dict = dict()
        for v in tf.trainable_variables('Generator//resnet_v2'):
            if 'resnet_v2_50/'+v.op.name[21:] in resnet_map.keys():
                resnet_dict['resnet_v2_50/'+v.op.name[21:]] = v
        resnet_var_name = [v.name for v in tf.trainable_variables('Generator//resnet_v2') \
                if 'resnet_v2_50/'+v.op.name[21:] in resnet_map.keys()]
        resnet_saver = tf.train.Saver(resnet_dict)

    saver = tf.train.Saver(save_vars, max_to_keep=100)
    branch_writers = [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "branch"+str(m))) \
        for m in range(FLAGS.num_branch)] #save generator loss for each branch 
    sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.checkpoint_dir, "CIS_Sum"),
                                 saver=None, save_summaries_secs=0)  

    with sv.managed_session() as sess:
        myprint ("Number of total params: {0} \n".format( \
            sess.run(total_parameter_count)))
        if FLAGS.resume_fullmodel:
            assert os.path.isfile(FLAGS.fullmodel_ckpt+'.index')
            saver.restore(sess, FLAGS.fullmodel_ckpt)
            myprint ("Resumed training from model {}".format(FLAGS.fullmodel_ckpt))
            myprint ("Start from step {}".format(sess.run(graph.global_step)))
            myprint ("Save checkpoint in          {}".format(FLAGS.checkpoint_dir))
            if not os.path.dirname(FLAGS.fullmodel_ckpt) == FLAGS.checkpoint_dir:
                print ("\033[0;30;41m"+"Warning: checkpoint dir and fullmodel ckpt do not match"+"\033[0m")
                myprint ("Please make sure that new checkpoint will be saved in the same dir with the resumed model")
        else:
            if FLAGS.resume_inpainter:
                assert os.path.isfile(FLAGS.inpainter_ckpt+'.index')  
                inpainter_saver.restore(sess, FLAGS.inpainter_ckpt)
                myprint ("Load pretrained inpainter {}".format(FLAGS.inpainter_ckpt))
                
            if FLAGS.resume_resnet:
                resnet_saver.restore(sess, FLAGS.resnet_ckpt)
                myprint ("Load pretrained resnet {}".format(FLAGS.resnet_ckpt))
            if not FLAGS.resume_resnet and not FLAGS.resume_inpainter:
                myprint ("Train from scratch")
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
                fetches["Inpainter_Loss"], fetches["Generator_Loss"] \
                    = graph.loss['Inpainter'], graph.loss['Generator']
                fetches["Inpainter_branch_Loss"], fetches["Generator_branch_Loss"] \
                    = graph.loss['Inpainter_branch'], graph.loss['Generator_branch'] 
                fetches['Generator_Loss_denominator'] = graph.loss['Generator_denominator']
                fetches['summary'] = summary_op

            if step % FLAGS.ckpt_steps == 0:
                fetches['generated_masks'] = graph.generated_masks
                fetches['GT_masks'] = graph.GT_masks

            results = sess.run(fetches, feed_dict={graph.is_training: True})
            progbar.update(step%FLAGS.ckpt_steps)

            if step % FLAGS.summaries_steps == 0 :
                print ("   Step:%3dk time:%4.4fmin   InpainterLoss%4.2f  GeneratorLoss%4.2f " \
                    %(step/1000, (time.time()-start_time)/60, results['Inpainter_Loss'], results['Generator_Loss']))
                sv.summary_writer.add_summary(results['summary'], step)

                generator_summary = sess.run(generator_summary_op,
                    feed_dict={graph.loss['Generator_var']: results['Generator_Loss']})
                sv.summary_writer.add_summary(generator_summary, step)

                for m in range(FLAGS.num_branch):
                    branch_summary = sess.run(branch_summary_op,
                        feed_dict={graph.loss['Inpainter_branch_var']: np.mean(results['Inpainter_branch_Loss'][:,m], axis=0),
                        graph.loss['Generator_branch_var']: np.mean(results['Generator_branch_Loss'][:,m], axis=0),
                        graph.loss['Generator_denominator_var']: np.mean(results['Generator_Loss_denominator'][:,m], axis=0)})
                    branch_writers[m].add_summary(branch_summary, step)              

            if step % FLAGS.ckpt_steps == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=step)
                progbar = Progbar(target=FLAGS.ckpt_steps)

                #evaluation
                sess.run(graph.val_iterator.initializer)
                fetches = {'GT_masks':graph.GT_masks, 'generated_masks':graph.generated_masks}

                num_sample = 200
                niter = num_sample//FLAGS.batch_size 
                score = 0
                for it in range(niter):
                    results_val = sess.run(fetches, feed_dict={graph.is_training:False})
                    for k in range(FLAGS.batch_size):
                        score += Permute_IoU(label=results_val['GT_masks'][k], pred=results_val['generated_masks'][k])
                score = score/num_sample      
                eval_summary = sess.run(eval_summary_op, feed_dict={graph.loss['EvalIoU_var']: score})
                sv.summary_writer.add_summary(eval_summary, step)                              
            step = results['step']

        myprint("Training completed")
