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
    # learner
    graph = Train_Graph(FLAGS)  
    graph.build()

    summary_op, tex_latent_summary_op, mask_latent_summary_op, bg_latent_summary_op = Summary.collect_VAE_summary(graph, FLAGS)
    # train
    #define model saver
    with tf.name_scope("parameter_count"):
        total_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                for v in tf.trainable_variables()])

    save_vars = tf.global_variables()
    
    if FLAGS.resume_CIS:
        CIS_vars = tf.global_variables('Inpainter')+tf.global_variables('Generator')
        CIS_saver = tf.train.Saver(CIS_vars, max_to_keep=100)

    saver = tf.train.Saver(save_vars, max_to_keep=100)
    tex_latent_writers = [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "tex_latent"+str(m))) for m in range(FLAGS.tex_dim)]
    bg_latent_writers = [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "bg_latent"+str(m))) for m in range(FLAGS.bg_dim)]
    mask_latent_writers =  [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "mask_latent"+str(m))) for m in range(FLAGS.mask_dim)]



    sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.checkpoint_dir, "VAE_Sum"),
                                 saver=None, save_summaries_secs=0)  #not saved automatically for flexibility

    with sv.managed_session() as sess:
        myprint ("Number of total params: {0} \n".format( \
            sess.run(total_parameter_count)))
        if FLAGS.resume_fullmodel:
            assert os.path.isfile(FLAGS.fullmodel_ckpt+'.index')
            saver.restore(sess, FLAGS.fullmodel_ckpt)
            myprint ("Resumed training from model {}".format(FLAGS.fullmodel_ckpt))
            myprint ("Start from vae_step{}".format(sess.run(graph.vae_global_step)))
            myprint ("Save checkpoint in          {}".format(FLAGS.checkpoint_dir))
            if not os.path.dirname(FLAGS.fullmodel_ckpt) == FLAGS.checkpoint_dir:
                print ("\033[0;30;41m"+"Warning: checkpoint dir and fullmodel ckpt do not match"+"\033[0m")
            #myprint ("Please make sure that the checkpoint will be saved in the same dir with the resumed model")
        else:
            if FLAGS.resume_CIS:
                assert os.path.isfile(FLAGS.CIS_ckpt+'.index')  
                CIS_saver.restore(sess, FLAGS.CIS_ckpt)
                myprint ("Load pretrained inpainter and generator {}".format(FLAGS.CIS_ckpt))
            else:
                myprint ("Train from scratch")
        myinput('Press enter to continue')

        start_time = time.time()
        #step = sess.run(graph.global_step)
        vae_step = sess.run(graph.vae_global_step)
        progbar = Progbar(target=FLAGS.ckpt_steps) #100k

        while (time.time()-start_time)<FLAGS.max_training_hrs*3600:
            if sv.should_stop():
                break

            fetches = {"vae_global_step_inc": graph.incr_vae_global_step, "vae_step": graph.vae_global_step}
            fetches['train_op'] = graph.train_ops
            mask_capacity = vae_step*FLAGS.mask_capacity_inc  #-> should have an VAE step

            if vae_step % FLAGS.summaries_steps == 0:
                fetches['tex_kl'], fetches['mask_kl'], fetches['bg_kl'] = graph.loss['tex_kl'], graph.loss['mask_kl'], graph.loss['bg_kl']
                fetches['Fusion'] = graph.loss['Fusion']
                fetches['summary'] = summary_op


            results = sess.run(fetches, feed_dict={graph.is_training: True, graph.mask_capacity: mask_capacity})
            progbar.update(vae_step%FLAGS.ckpt_steps)

            if vae_step % FLAGS.summaries_steps == 0 :
                print ("   Step:%3dk time:%4.4fmin " \
                    %(vae_step/1000, (time.time()-start_time)/60))
                sv.summary_writer.add_summary(results['summary'], vae_step)

                for d in range(FLAGS.tex_dim):
                    tex_summary = sess.run(tex_latent_summary_op, feed_dict={graph.loss['tex_kl_var']: results['tex_kl'][d]})
                    tex_latent_writers[d].add_summary(tex_summary, vae_step)
                    
                for d in range(FLAGS.bg_dim):
                    bg_summary = sess.run(bg_latent_summary_op, feed_dict={graph.loss['bg_kl_var']: results['bg_kl'][d]})
                    bg_latent_writers[d].add_summary(bg_summary, vae_step)

                for d in range(FLAGS.mask_dim):
                    mask_summary = sess.run(mask_latent_summary_op, feed_dict={graph.loss['mask_kl_var']: results['mask_kl'][d]})
                    mask_latent_writers[d].add_summary(mask_summary, vae_step)
                
              
            if vae_step % FLAGS.ckpt_steps == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=vae_step)
                progbar = Progbar(target=FLAGS.ckpt_steps)

            vae_step = results['vae_step']

        myprint("Training completed")