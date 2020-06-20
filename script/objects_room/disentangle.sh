ABSPATH=$(readlink -f $0)
CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH \
--checkpoint_dir=outputs/objects_room/tex_split \
--dataset=objects_room  --num_branch=6 --root_dir=data/objects_room_data/objects_room_train.tfrecords \
--mode=eval_VAE --model=resnet_v2_50 \
\
\
--fullmodel_ckpt='path to model checkpoint' \
--tex_dim=5 --bg_dim=10 --mask_dim=10 \
\
--input_img=sample_imgs/objects_room/01.png \
--traverse_type=tex --top_kdim=1 --traverse_branch=0,2 --traverse_start=-0.5 --traverse_end=1.5 \
--batch_size=1 \

#checkpoint_dir:  folder containing outputs

#input_img: target image (only support processing single image)

#traverse_type: 
        #mask: traverse shape latent space
        #tex: traverse texture/color latent space
        #bg: traverse background latent space

#top_kdim: 
    #choose k dimensions with the largest kl divergence to traverse
    #these dimensions should encode k most significant variables of the object.
    #results of traversing dimension of kth largest kl divergence are output as branch{i}_var{k}.gif


#traverse_branch: which branches to traverse  
    #(only effective when traverse_type in ['tex','mask'])
    #'all': generate results of traversing all branches except the background branch
    #'0,1,2': mannually choose branches to traverse