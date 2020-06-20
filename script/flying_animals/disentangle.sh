ABSPATH=$(readlink -f $0)


CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH \
--checkpoint_dir=outputs/flying_animals/02_bg_range6 \
--dataset=flying_animals  --max_num=5  --num_branch=6 --root_dir=data/flying_animals_data/img_data.npz \
--mode=eval_VAE --model=resnet_v2_50 \
\
\
--fullmodel_ckpt='path to checkpoint' \
--tex_dim=20 --bg_dim=30 --mask_dim=20 \
\
--input_img=sample_imgs/flying_animals/02.png \
--traverse_type=bg --top_kdim=4 --traverse_branch=5 \
--batch_size=1 --traverse_start=-6 --traverse_end=6 \

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
