ABSPATH=$(readlink -f $0)


CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH  \
--checkpoint_dir=checkpoint/multi_dsprites/VAE \
--dataset=multi_dsprites --num_branch=5 --root_dir=data/multi_dsprites_data/multi_dsprites_colored_on_colored.tfrecords \
--mode=train_VAE --model=resnet_v2_50 \
\
\
--resume_CIS=True --resume_fullmodel=False \
--fullmodel_ckpt= \
--CIS_ckpt='path of CIS ckpt to resume here, e.g. checkpoint/multi_dsprites/CIS/model-100000' \
\
--batch_size=16 --VAE_lr=1e-4 \
--tex_dim=5 --tex_beta=6 \
--bg_dim=5 --bg_beta=6 \
--mask_dim=10 --mask_gamma=500 --mask_capacity_inc=2e-5 \
--ckpt_steps=20000 --summaries_steps=1000 \

