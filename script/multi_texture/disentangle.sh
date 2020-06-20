ABSPATH=$(readlink -f $0)

CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH \
--checkpoint_dir=outputs/multi_texture \
--dataset=multi_texture --max_num=4 --num_branch=5 --root_dir=data/multi_texture_data/ \
--mode=eval_VAE --model=resnet_v2_50 \
\
\
--fullmodel_ckpt='path to checkpoint' \
--tex_dim=5 --mask_dim=10 --bg_dim=5 \
\
--input_img=sample_imgs/multi_texture/01.png \
--traverse_type=mask --top_kdim=2 --traverse_branch=0,1,2,3 --traverse_range=0.5 \
--batch_size=1 \

#checkpoint_dir:  folder containing outputs

#input_img: target image (only support processing single image)

#traverse_type: 
        #mask: traverse shape latent space
        #tex: traverse texture/color latent space
        #bg: traverse background latent space

#top_kdim: 
    #choose k dimensions with largest kl divergence to traverse
    #these dimensions should encode k most significant variables of the object.
    #results of traversing dimension of kth largest kl divergence are output as branch{i}_var{k}.gif


#traverse_branch: which branches to traverse  
    #(only effective when traverse_type in ['tex','mask'])
    #'all': generate results of traversing all branches except the background branch
    #'0,1,2': mannually choose branches to traverse
