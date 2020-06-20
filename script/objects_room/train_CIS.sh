ABSPATH=$(readlink -f $0)

CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH  \
--checkpoint_dir=checkpoint/objects_room/CIS \
--dataset=objects_room  --num_branch=6 --root_dir=data/objects_room_data/objects_room_train.tfrecords \
--mode=train_CIS --model=resnet_v2_50 \
\
\
--resume_inpainter=True --resume_fullmodel=False \
--fullmodel_ckpt=/  \
--inpainter_ckpt='path of pretrained inpainter ckpt to resume here'
                'e.g. checkpoint/objects_room/pretrain_inpainter/inpainter-50000' \
\
--batch_size=8 --inp_lr=1e-4 --gen_lr=3e-5 \
--epsilon=30 --iters_inp=1 --iters_gen=3 \
--ckpt_steps=4000 \