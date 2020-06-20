ABSPATH=$(readlink -f $0)

CUDA_VISIBLE_DEVICES=0 python main.py  \
--checkpoint_dir=checkpoint/objects_room/pretrain_inpainter \
--sh_path=$ABSPATH \
 \
--dataset=objects_room  --root_dir=data/objects_room_data/objects_room_train.tfrecords \
--mode=pretrain_inpainter \
 \
 --batch_size=16 --inp_lr=3e-5 \
 \
 --summaries_secs=60 --ckpt_secs=10000 \

