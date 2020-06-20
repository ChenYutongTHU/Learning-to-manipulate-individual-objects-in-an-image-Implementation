ABSPATH=$(readlink -f $0)


CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH \
--checkpoint_dir=checkpoint/objects_room/VAE \
--dataset=objects_room  --num_branch=6 --root_dir=data/objects_room_data/objects_room_train.tfrecords \
--mode=train_VAE --model=resnet_v2_50 \
\
\
--resume_CIS=True --resume_fullmodel=False \
--fullmodel_ckpt=/ \
--CIS_ckpt='path of CIS ckpt to resume here, e.g. checkpoint/objects_room/CIS/model-100000' \
\
--batch_size=8 --VAE_lr=1e-4 \
--tex_dim=5 --tex_beta=6 \
--bg_dim=5 --bg_beta=6 \
--mask_dim=10 --mask_gamma=500 --mask_capacity_inc=2e-4 \
--ckpt_steps=20000 --summaries_steps=1000 \