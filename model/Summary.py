import tensorflow as tf

def convert2uint8(img_list):
     #float[0~1] -> uint8[0~255]
    new_img_list = [tf.cast(img*255, tf.uint8) for img in img_list]
    return new_img_list

def collect_globalVAE_summary(graph, FLAGS):
    ori = graph.image_batch[0]
    reconstr = graph.out_imgs[0]
    show_list = convert2uint8([ori, reconstr])
    tf.compat.v1.summary.image('image output', tf.stack(show_list, axis=0), max_outputs=len(show_list), collections=["globalVAE_Sum"])

    tf.summary.scalar('Reconstruction_Loss', graph.loss, collections=["globalVAE_Sum"])
    tf.summary.scalar('latent_space', graph.kl_var, collections=['globalVAE_Sum_kl'])

    for grad, var in graph.train_vars_grads:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['globalVAE_Sum'])

    return tf.summary.merge(tf.compat.v1.get_collection("globalVAE_Sum")), \
        tf.summary.merge(tf.compat.v1.get_collection("globalVAE_Sum_kl"))


def collect_CIS_summary(graph, FLAGS):
    #----image to show same with Inpainter_Sum----
    ori = graph.image_batch[0] 
    edge = tf.concat([graph.edge_map[0], tf.zeros_like(graph.edge_map[0,:,:,0:1])], axis=-1) #H W 3
    mean = graph.unconditioned_mean[0]
    show_list = convert2uint8([ori, edge,mean]) #0~255
    #show_list = [tf.cast(edge*128, tf.uint8)]
    tf.compat.v1.summary.image('image_edge_unconditionedMean', 
        tf.stack(show_list, axis=0), max_outputs=len(show_list), 
        collections=["CIS_Sum"])

    for i in range(FLAGS.num_branch):
        mask = graph.generated_masks[0,:,:,:,i]
        #aug = mask*ori + ori*(1-mask)*0.2 
        context = ori *(1-mask)  # H W 3
        GT = ori*mask
        predict = graph.pred_intensities[0,:,:,:,i]*mask
        show_list = convert2uint8([GT,context,predict])
        tf.compat.v1.summary.image('branch{}'.format(i), 
            tf.stack(show_list, axis=0), max_outputs=len(show_list), 
            collections=["CIS_Sum"])

    #-----curve to show
    #plot multiple curvations in one figure
    tf.summary.scalar('Inpainter_Loss', graph.loss['Inpainter'], collections=['CIS_Sum'])
    tf.summary.scalar('Generator_Loss', graph.loss['Generator_var'], collections=['CIS_Sum_Generator'])
    tf.summary.scalar('Inpainter_Loss/ branch', graph.loss['Inpainter_branch_var'], collections=['CIS_Sum_branch'])
    tf.summary.scalar('Generator_Loss/ branch', graph.loss['Generator_branch_var'], collections=['CIS_Sum_branch'])
    tf.summary.scalar('Generator_Loss/ denominator', graph.loss['Generator_denominator_var'], collections=['CIS_Sum_branch'])
    tf.summary.scalar('IoU Validation',graph.loss['EvalIoU_var'], collections=['CIS_eval'])

    #------histogram to show
    for grad, var in graph.train_vars_grads['Generator']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['CIS_Sum'])
    for grad, var in graph.train_vars_grads['Inpainter']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['CIS_Sum'])   
    return tf.summary.merge(tf.compat.v1.get_collection("CIS_Sum")), \
    tf.summary.merge(tf.compat.v1.get_collection("CIS_Sum_Generator")), \
    tf.summary.merge(tf.compat.v1.get_collection("CIS_Sum_branch")), \
    tf.summary.merge(tf.compat.v1.get_collection("CIS_eval"))


def collect_VAE_summary(graph, FLAGS):
    ori = graph.image_batch[0] 
    fusion = graph.fusion_outputs[0]
    show_list = convert2uint8([ori, fusion])
    tf.compat.v1.summary.image('image output', tf.stack(show_list, axis=0), max_outputs=len(show_list), collections=["VAE_Sum"])

    seg_masks = tf.transpose(graph.generated_masks[0,:,:,:,:]*tf.expand_dims(ori, axis=-1),[3,0,1,2]) #N H W 3
    tf.compat.v1.summary.image('segmentation', tf.cast(seg_masks*255,tf.uint8), max_outputs=FLAGS.num_branch, collections=["VAE_Sum"])

    for i in range(FLAGS.num_branch-FLAGS.n_bg):
        seg = graph.generated_masks[0,:,:,:,i]*ori
        out_mask = tf.tile(graph.VAE_outmasks[0,:,:,:,i],[1,1,3])
        out_tex = graph.VAE_outtexes[0,:,:,:,i]
        fusion = graph.VAE_fusion[0,:,:,:,i]
        show_list = convert2uint8([seg, out_mask, out_tex, fusion])
        tf.compat.v1.summary.image('branch{}'.format(i), tf.stack(show_list, axis=0), max_outputs=len(show_list), collections=["VAE_Sum"])  

    #background
    seg = tf.reduce_sum(graph.generated_masks[0,:,:,:,-1*FLAGS.n_bg:], axis=-1)*ori
    out_bg_tex = graph.VAE_outtex_bg[0,:,:,:] #H W 3
    show_list = convert2uint8([seg, out_bg_tex])
    tf.compat.v1.summary.image('background', tf.stack(show_list, axis=0), max_outputs=len(show_list), collections=["VAE_Sum"]) 

    tf.summary.scalar('Tex_error', graph.loss['tex_error'], collections=["VAE_Sum"])
    tf.summary.scalar('Mask_error', graph.loss['mask_error'], collections=["VAE_Sum"])
    tf.summary.scalar('BG_error', graph.loss['bg_error'], collections=["VAE_Sum"])

    tf.summary.scalar('VAEFusion_error', graph.loss['VAE_fusion_error'], collections=["VAE_Sum"])
    tf.summary.scalar('Fusion_Loss', graph.loss['Fusion'], collections=["VAE_Sum"])

    tf.summary.scalar('Tex_latent_space', graph.loss['tex_kl_var'], collections=['VAE_Sum_tex'])
    tf.summary.scalar('Mask_latent_space', graph.loss['mask_kl_var'], collections=['VAE_Sum_mask'])
    tf.summary.scalar('BG_latent_space', graph.loss['bg_kl_var'], collections=['VAE_Sum_bg'])

    for grad, var in graph.train_vars_grads['VAE//separate']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['VAE_Sum'])
    for grad, var in graph.train_vars_grads['VAE//fusion']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['VAE_Sum'])
    for grad, var in graph.train_vars_grads['Fusion']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['VAE_Sum'])

    return tf.summary.merge(tf.compat.v1.get_collection("VAE_Sum")), \
        tf.summary.merge(tf.compat.v1.get_collection("VAE_Sum_tex")), \
        tf.summary.merge(tf.compat.v1.get_collection("VAE_Sum_mask")), \
        tf.summary.merge(tf.compat.v1.get_collection("VAE_Sum_bg"))

def collect_end2end_summary(graph, FLAGS):
    ori = graph.image_batch[0] 
    fusion = graph.fusion_outputs[0]
    show_list = convert2uint8([ori, fusion])
    tf.compat.v1.summary.image('image output', tf.stack(show_list, axis=0), max_outputs=len(show_list), collections=["end2end_Sum"])

    seg_masks = tf.transpose(graph.generated_masks[0,:,:,:,:]*tf.expand_dims(ori, axis=-1),[3,0,1,2]) #N H W 3
    tf.compat.v1.summary.image('segmentation', tf.cast(seg_masks*255,tf.uint8), max_outputs=FLAGS.num_branch, collections=["end2end_Sum"])
    
    for i in range(FLAGS.num_branch-FLAGS.n_bg):
        seg = graph.generated_masks[0,:,:,:,i]*ori
        out_mask = tf.tile(graph.VAE_outmasks[0,:,:,:,i],[1,1,3])
        out_tex = graph.VAE_outtexes[0,:,:,:,i]
        fusion = graph.VAE_fusion[0,:,:,:,i]
        show_list = convert2uint8([seg, out_mask, out_tex, fusion])
        tf.compat.v1.summary.image('branch{}'.format(i), tf.stack(show_list, axis=0), max_outputs=len(show_list), collections=["end2end_Sum"])

    #background
    seg = tf.reduce_sum(graph.generated_masks[0,:,:,:,-1*FLAGS.n_bg:], axis=-1)*ori
    out_bg_tex = graph.VAE_outtex_bg[0,:,:,:] #H W 3
    show_list = convert2uint8([seg, out_bg_tex])
    tf.compat.v1.summary.image('background', tf.stack(show_list, axis=0), max_outputs=len(show_list), collections=["end2end_Sum"])


    #-----curve to show-------------
    #tf.summary.scalar('CIS', graph.loss['CIS'], collections=['end2end_Sum'])
    #tf.summary.scalar('Inpainter_Loss', graph.loss['Inpainter'], collections=['end2end_Sum'])
    tf.summary.scalar('Tex_error', graph.loss['tex_error'], collections=["end2end_Sum"])
    #tf.summary.scalar('Mask_error', graph.loss['mask_error'], collections=["end2end_Sum"])
    tf.summary.scalar('BG_error', graph.loss['bg_error'], collections=["end2end_Sum"])

    tf.summary.scalar('VAEFusion_error', graph.loss['VAE_fusion_error'], collections=["end2end_Sum"])
    tf.summary.scalar('Fusion_Loss', graph.loss['Fusion'], collections=["end2end_Sum"])

    tf.summary.scalar('Tex_latent_space', graph.loss['tex_kl_var'], collections=['end2end_Sum_tex'])
    #tf.summary.scalar('Mask_latent_space', graph.loss['mask_kl_var'], collections=['end2end_Sum_mask'])
    tf.summary.scalar('BG_latent_space', graph.loss['bg_kl_var'], collections=['end2end_Sum_bg'])

    tf.summary.scalar('IoU Validation',graph.loss['EvalIoU_var'], collections=['CIS_eval'])

    #-----histogram to show-----------
    for grad, var in graph.train_vars_grads['VAE//separate/texVAE']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['end2end_Sum'])
    for grad, var in graph.train_vars_grads['VAE//separate/bgVAE']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['end2end_Sum'])
    for grad, var in graph.train_vars_grads['VAE//fusion']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['end2end_Sum'])
    for grad, var in graph.train_vars_grads['Fusion']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['end2end_Sum'])

    return tf.summary.merge(tf.compat.v1.get_collection("end2end_Sum")), \
        tf.summary.merge(tf.compat.v1.get_collection("end2end_Sum_tex")), \
        tf.summary.merge(tf.compat.v1.get_collection("end2end_Sum_bg")), \
        tf.summary.merge(tf.compat.v1.get_collection("CIS_eval"))

   

def collect_inpainter_summary(graph, FLAGS):
    #---------image to show-------------

    # original image  edge_map
    ori = graph.image_batch[0] 
    edge = tf.concat([graph.edge_map[0], tf.zeros_like(graph.edge_map[0,:,:,0:1])], axis=-1) #H W 3
    show_list = convert2uint8([ori, edge, graph.unconditioned_mean[0]]) #0~255

    tf.compat.v1.summary.image('image_edge', 
        tf.stack(show_list, axis=0), max_outputs=len(show_list), 
        collections=["Inpainter_Sum"])

    for i in range(FLAGS.num_branch):
        mask = graph.generated_masks[0,:,:,:,i]
        context = ori *(1-mask)  
        GT = ori*mask
        predict = graph.pred_intensities[0,:,:,:,i]*mask
        show_list = convert2uint8([GT,context,predict])
        tf.compat.v1.summary.image('branch{}'.format(i), 
            tf.stack(show_list, axis=0), max_outputs=len(show_list), 
            collections=["Inpainter_Sum"])

    loss = graph.loss['Inpainter']
    tf.summary.scalar('Inpainter_Loss', loss, collections=['Inpainter_Sum'])

    for grad, var in graph.train_vars_grads['Inpainter']:
        tf.summary.histogram(var.op.name+'/grad', grad, collections=['Inpainter_Sum'])

    return tf.summary.merge(tf.compat.v1.get_collection('Inpainter_Sum'))