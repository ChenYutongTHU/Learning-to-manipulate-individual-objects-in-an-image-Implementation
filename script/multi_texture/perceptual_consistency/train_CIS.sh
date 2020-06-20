ABSPATH=$(readlink -f $0)

CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH \
--checkpoint_dir=checkpoint/multi_texture/CIS/PC/CIS \
--dataset=multi_texture --max_num=2 --num_branch=3  --root_dir=data/multi_texture_data/ \
--mode=train_CIS --model=segnet --PC=True \
\
\
--resume_inpainter=True --resume_fullmodel=False \
--fullmodel_ckpt= \
--inpainter_ckpt='path of pretrained inpainter ckpt to resume here'
                'e.g. checkpoint/multi_texture/pretrain_inpainter/inpainter-50000' \
\
--batch_size=8 --inp_lr=1e-4 --gen_lr=3e-5 \
--epsilon=50 --iters_inp=1 --iters_gen=3 \
--ckpt_steps=5000 \