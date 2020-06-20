ABSPATH=$(readlink -f $0)


CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH  \
--checkpoint_dir=checkpoint/multi_texture/CIS/PC/VAE \
--dataset=multi_texture --max_num=2 --num_branch=3 --root_dir=data/multi_texture_data/ --PC=True \
--mode=train_VAE --model=segnet \
\
\
--resume_CIS=True --resume_fullmodel=False \
--fullmodel_ckpt= \
--CIS_ckpt='path of CIS ckpt to resume here' \
\
--batch_size=16 --VAE_lr=1e-4 \
--tex_dim=5 --tex_beta=4 \
--bg_dim=5 --bg_beta=6 \
--mask_dim=10 --mask_gamma=500 --mask_capacity_inc=2e-5 \
--ckpt_steps=20000 --summaries_steps=1000 \