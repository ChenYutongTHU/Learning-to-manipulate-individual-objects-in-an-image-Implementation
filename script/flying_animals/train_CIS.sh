ABSPATH=$(readlink -f $0)

CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH \
--checkpoint_dir=checkpoint/flying_animals/CIS \
--dataset=flying_animals --max_num=5  --num_branch=6 --root_dir=data/flying_animals_data/img_data.npz \
--mode=train_CIS --model=resnet_v2_50 \
\
\
--resume_inpainter=True --resume_fullmodel=False --resume_resnet=True \
--fullmodel_ckpt=/  \
--inpainter_ckpt='path of pretrained inpainter ckpt to resume here'
                'e.g. checkpoint/flying_animals/pretrain_inpainter/inpainter-100000' \
--resnet_ckpt=resnet/resnet_v2_50/resnet_v2_50.ckpt \
\
--batch_size=4 --inp_lr=1e-4 --gen_lr=1e-4 \
--epsilon=100 --iters_inp=1 --iters_gen=2 \
--ckpt_steps=10000 --summaries_steps=500 \