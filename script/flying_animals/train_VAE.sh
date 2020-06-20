ABSPATH=$(readlink -f $0)

CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH \
--checkpoint_dir=checkpoint/flying_animals/VAE \
--dataset=flying_animals  --max_num=5  --num_branch=6 --root_dir=data/flying_animals_data/img_data.npz \
--mode=train_VAE --model=resnet_v2_50 \
\
\
--resume_CIS=True --resume_fullmodel=False \
--fullmodel_ckpt=/ \
--CIS_ckpt='path of CIS ckpt to resume here, e.g. checkpoint/flying_animals/CIS/model-100000' \
\
--batch_size=4 --VAE_lr=1e-4 \
--tex_dim=20 --tex_beta=4 \
--bg_dim=30 --bg_beta=4 \
--mask_dim=20 --mask_gamma=500 --mask_capacity_inc=5e-5 \
--ckpt_steps=10000 --summaries_steps=1000 \