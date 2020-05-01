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

    summary_op, tex_latent_summary_op, bg_latent_summary_op, eval_summary_op = Summary.collect_end2end_summary(graph, FLAGS)
    # train
    #define model saver
    with tf.name_scope("parameter_count"):
        total_parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                for v in tf.trainable_variables()])

    save_vars = tf.global_variables()
    # tf.global_variables('Inpainter')+tf.global_variables('Generator')+ \
    #     tf.global_variables('VAE')+tf.global_variables('Fusion') \
    #     +tf.global_variables('train_op') #including global step
    
    if FLAGS.resume_CIS:
        CIS_vars = tf.global_variables('Inpainter')+tf.global_variables('Generator')
        CIS_saver = tf.train.Saver(CIS_vars, max_to_keep=100)

    mask_saver = tf.train.Saver(tf.global_variables('VAE//separate/maskVAE/'), max_to_keep=100)
    tex_saver = tf.train.Saver(tf.global_variables('VAE//separate/texVAE/'), max_to_keep=100)

    saver = tf.train.Saver(save_vars, max_to_keep=100)
    branch_writers = [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir,'branch'+str(m))) for m in range(FLAGS.num_branch)]
    tex_latent_writers = [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "tex_latent"+str(m))) for m in range(FLAGS.tex_dim)]
    bg_latent_writers = [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "bg_latent"+str(m))) for m in range(FLAGS.bg_dim)]
    #mask_latent_writers =  [tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, "mask_latent"+str(m))) for m in range(FLAGS.mask_dim)]



    sv = tf.train.Supervisor(logdir=os.path.join(FLAGS.checkpoint_dir, "end2end_Sum"),
                                 saver=None, save_summaries_secs=0)  #not saved automatically for flexibility

    with sv.managed_session() as sess:
        myprint ("Number of total params: {0} \n".format( \
            sess.run(total_parameter_count)))
        if FLAGS.resume_fullmodel:
            assert os.path.isfile(FLAGS.fullmodel_ckpt+'.index')
            saver.restore(sess, FLAGS.fullmodel_ckpt)
            myprint ("Resumed training from model {}".format(FLAGS.fullmodel_ckpt))
            myprint ("Start from step {} vae_step{}".format(sess.run(graph.global_step), sess.run(graph.vae_global_step)))
            myprint ("Save checkpoint in          {}".format(FLAGS.checkpoint_dir))
            if not os.path.dirname(FLAGS.fullmodel_ckpt) == FLAGS.checkpoint_dir:
                print ("\033[0;30;41m"+"Warning: checkpoint dir and fullmodel ckpt do not match"+"\033[0m")
            #myprint ("Please make sure that the checkpoint will be saved in the same dir with the resumed model")
        else:
            if os.path.isfile(FLAGS.mask_ckpt+'.index'):
                mask_saver.restore(sess, FLAGS.mask_ckpt)
                myprint ("Load pretrained maskVAE {}".format(FLAGS.mask_ckpt))
            if os.path.isfile(FLAGS.tex_ckpt+'.index'):   
                tex_saver.restore(sess, FLAGS.tex_ckpt)
                myprint ("Load pretrained texVAE {}".format(FLAGS.tex_ckpt))
            if FLAGS.resume_CIS:
                assert os.path.isfile(FLAGS.CIS_ckpt+'.index')  
                CIS_saver.restore(sess, FLAGS.CIS_ckpt)
                myprint ("Load pretrained inpainter and generator {}".format(FLAGS.CIS_ckpt))
            else:
                myprint ("Train from scratch")
        myinput('Press enter to continue')

        start_time = time.time()
        step = sess.run(graph.global_step)
        vae_step = sess.run(graph.vae_global_step)
        progbar = Progbar(target=FLAGS.ckpt_steps) #100k

        sum_iters = FLAGS.iters_gen_vae + FLAGS.iters_inp

        while (time.time()-start_time)<FLAGS.max_training_hrs*3600:
            if sv.should_stop():
                break

            fetches = {"global_step_inc": graph.incr_global_step, "step": graph.global_step}

            if step%sum_iters < FLAGS.iters_inp:
                fetches['train_op'] = graph.train_ops['Inpainter']
                mask_capacity = vae_step*FLAGS.mask_capacity_inc
            else:
                fetches['train_op'] = graph.train_ops #'VAE//separate/texVAE/','VAE//separate/texVAE_BG/', 'VAE//fusion', 'Fusion'
                mask_capacity = vae_step*FLAGS.mask_capacity_inc  #-> should have an VAE step
                fetches['vae_global_step'], fetches['vae_global_step_inc'] = graph.vae_global_step, graph.incr_vae_global_step

            if step % FLAGS.summaries_steps == 0:
                fetches["Inpainter_Loss"],fetches["Generator_Loss"] = graph.loss['Inpainter'], graph.loss['Generator']
                fetches["VAE//texVAE"], fetches["VAE//texVAE_BG"], fetches['VAE//fusion'] = graph.loss['VAE//separate/texVAE/'], graph.loss['VAE//separate/texVAE_BG/'], graph.loss['VAE//fusion']
                fetches['tex_kl'], fetches['bg_kl'] = graph.loss['tex_kl'], graph.loss['bg_kl']
                fetches['summary'] = summary_op

            if step % FLAGS.ckpt_steps == 0:
                fetches['generated_masks'] = graph.generated_masks
                fetches['GT_masks'] = graph.GT_masks

            results = sess.run(fetches, feed_dict={graph.is_training: True, graph.mask_capacity: mask_capacity})
            progbar.update(step%FLAGS.ckpt_steps)

            if step % FLAGS.summaries_steps == 0 :
                print ("   Step:%3dk time:%4.4fmin   VAELoss%4.2f" \
                    %(step/1000, (time.time()-start_time)/60, results["VAE//texVAE"]+results['VAE//fusion']+results['VAE//texVAE_BG']))
                sv.summary_writer.add_summary(results['summary'], step)

                for d in range(FLAGS.tex_dim):
                    tex_summary = sess.run(tex_latent_summary_op, feed_dict={graph.loss['tex_kl_var']: results['tex_kl'][d]})
                    tex_latent_writers[d].add_summary(tex_summary, step)
                    
                for d in range(FLAGS.bg_dim):
                    bg_summary = sess.run(bg_latent_summary_op, feed_dict={graph.loss['bg_kl_var']: results['bg_kl'][d]})
                    bg_latent_writers[d].add_summary(bg_summary, step)

                # for d in range(FLAGS.mask_dim):
                #     mask_summary = sess.run(mask_latent_summary_op, feed_dict={graph.loss['mask_kl_var']: results['mask_kl'][d]})
                #     mask_latent_writers[d].add_summary(mask_summary, step)
                
              
            if step % FLAGS.ckpt_steps == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=step)
                progbar = Progbar(target=FLAGS.ckpt_steps)

                #evaluation
                sess.run(graph.val_iterator.initializer)
                fetches = {'GT_masks':graph.GT_masks, 'generated_masks':graph.generated_masks}

                if FLAGS.dataset in ['multi_texture', 'flying_animals']:
                    #note that for multi_texture bg_num is just a fake number it represents number of samples for each type of image
                    score = [[]]*FLAGS.max_num
                    for bg in range(FLAGS.bg_num):
                        results_val=sess.run(fetches, feed_dict={graph.is_training: False})
                        for k in range(FLAGS.max_num):
                            #score[k].append(Permute_IoU(results_val['GT_masks'][k], results_val['generated_masks'][k]))
                            score[k] = score[k] + [Permute_IoU(label=results_val['GT_masks'][k], pred=results_val['generated_masks'][k])]
                    for k in range(FLAGS.max_num):
                        eval_summary = sess.run(eval_summary_op, feed_dict={graph.loss['EvalIoU_var']: np.mean(score[k])})
                        branch_writers[k+1].add_summary(eval_summary, step)
                else:
                    num_sample = FLAGS.skipnum
                    niter = num_sample//FLAGS.batch_size
                    assert num_sample%FLAGS.batch_size==0
                    score = 0
                    for it in range(niter):
                        results_val = sess.run(fetches, feed_dict={graph.is_training:False})
                        for k in range(FLAGS.batch_size):
                            score += Permute_IoU(label=results_val['GT_masks'][k], pred=results_val['generated_masks'][k])
                    score = score/num_sample
                    eval_summary = sess.run(eval_summary_op, feed_dict={graph.loss['EvalIoU_var']: score})
                    sv.summary_writer.add_summary(eval_summary, step)

            step = results['step']
            vae_step = results['vae_global_step']

        myprint("Training completed")