ABSPATH=$(readlink -f $0)

CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH  \
--checkpoint_dir=checkpoint/multi_texture/CIS/PC/finetune \
--dataset=multi_texture --max_num=2 --num_branch=3  --root_dir=data/multi_texture_data/ \
--mode=train_PC --model=segnet --PC=True \
\
--fullmodel_ckpt='path of pretrained ckpt to finetune here'
                'e.g. checkpoint/multi_texture/pc/model-0'  \
--batch_size=4 --inp_lr=1e-4 --gen_lr=3e-5  \
--bg_dim=5 --tex_dim=5 --mask_dim=10 \
--epsilon=50 --ita=1e-3 --iters_inp=1 --iters_gen_vae=3 \
--ckpt_steps=10000 --summaries_steps=100 \
