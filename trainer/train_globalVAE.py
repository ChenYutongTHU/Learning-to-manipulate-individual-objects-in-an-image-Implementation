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
from model.globalVAE_graph import Train_Graph
import imageio
import numpy as np
import time
import re

def train(FLAGS):
    # learner
    graph = Train_Graph(FLAGS)  
    graph.build()

    summary_op, latent_summary_op = Summary.collect_globalVAE_summary(graph, FLAGS)
    # train
    #define model saver
    with tf.name_scope("parameter_count"):
        total_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                for v in tf.trainable_variables()])

    save_vars = tf.global_variables()
    saver = tf.train.Saver(save_vars, max_to_keep=100)

    latent_writers = [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "latent"+str(m))) \
        for m in range(FLAGS.tex_dim)] 
    sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.checkpoint_dir, "globalVAE_Sum"),
                                 saver=None, save_summaries_secs=0)  #not saved automatically for flexibility

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
            #myprint ("Please make sure that the checkpoint will be saved in the same dir with the resumed model")
        else:
            myprint ("Train from scratch")
        myinput('Press enter to continue')

        start_time = time.time()
        step = sess.run(graph.global_step)
        progbar = Progbar(target=FLAGS.ckpt_steps) #100k

        while (time.time()-start_time)<FLAGS.max_training_hrs*3600:
            if sv.should_stop():
                break

            fetches = {"global_step_inc": graph.incr_global_step, "step": graph.global_step, "train_op": graph.train_ops}

            if step % FLAGS.summaries_steps == 0:
                fetches["Loss"] = graph.loss
                fetches["kl_dim"] = graph.latent_loss_dim #dim,
                fetches['summary'] = summary_op

            results = sess.run(fetches)
            progbar.update(step%FLAGS.ckpt_steps)

            if step % FLAGS.summaries_steps == 0 :
                print ("   Step:%3dk time:%4.4fmin   Loss%4.2f  " \
                    %(step/1000, (time.time()-start_time)/60, results['Loss']))
                sv.summary_writer.add_summary(results['summary'], step)

                for m in range(FLAGS.tex_dim):
                    kl_summary = sess.run(latent_summary_op,
                        feed_dict={graph.kl_var: results['kl_dim'][m]})
                    latent_writers[m].add_summary(kl_summary, step)

        

            if step % FLAGS.ckpt_steps == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=step)
                progbar = Progbar(target=FLAGS.ckpt_steps)

            step = results['step']

        myprint("Training completed")
