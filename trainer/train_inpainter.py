import tensorflow as tf 
import os
import gflags  
#https://github.com/google/python-gflags
import sys
#sys.path.append("..") 
import pprint
from keras.utils.generic_utils import Progbar
import model.Summary as Summary
from model.utils.generic_utils import myprint, myinput
from model.train_graph import Train_Graph
import imageio
import numpy as np
import time
def train(FLAGS):
    graph = Train_Graph(FLAGS)  
    graph.build()

    summary_op = Summary.collect_inpainter_summary(graph, FLAGS)

    with tf.name_scope("parameter_count"):
        total_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                for v in tf.trainable_variables()])
        inpainter_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                for v in tf.trainable_variables(scope='Inpainter')])
    save_vars = tf.global_variables('Inpainter')+tf.global_variables('train_op')+tf.global_variables('summary_vars')
    saver = tf.train.Saver(save_vars, max_to_keep=100)
    sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.checkpoint_dir, "Inpainter_Sum"),
                                 global_step=graph.global_step,
                                 saver=saver, checkpoint_basename='Inpainter', save_model_secs=FLAGS.ckpt_secs, 
                                 summary_op=summary_op, #summary_writer=USE_DEFAULT, 
                                 save_summaries_secs=FLAGS.summaries_secs)

    with sv.managed_session() as sess:
        myprint ("Number of total params: {0} \n".format( \
            sess.run(total_parameter_count)))
        start_time = time.time()
        step = sess.run(graph.global_step)
        progbar = Progbar(target=100000) #100k
        while (time.time()-start_time)<FLAGS.max_training_hrs*3600:
            if sv.should_stop():
                break
            fetches = {
                "train_op":graph.train_ops['Inpainter'],
                "loss": graph.loss['Inpainter'],
                "global_step_inc": graph.incr_global_step
            }
            results = sess.run(fetches, feed_dict={graph.is_training: True})

            if step%1000 == 0:
                print ("   Step:%3dk time:%4.4fmin   InpainterLoss%4.2f "%(step/1000, 
                    (time.time()-start_time)/60, results['loss']))

            if step % 100000 == 0:
                progbar = Progbar(target=100000)
            progbar.update(step%100000)
            step += 1

        myprint("Training completed")




    