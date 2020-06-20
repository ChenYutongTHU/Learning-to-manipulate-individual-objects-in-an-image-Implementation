ABSPATH=$(readlink -f $0)


CUDA_VISIBLE_DEVICES=0 python main.py  \
--checkpoint_dir=checkpoint/multi_dsprites/pretrain_inpainter \
--sh_path=$ABSPATH \
 \
--dataset=multi_dsprites  --root_dir=data/multi_dsprites_data/multi_dsprites_colored_on_colored.tfrecords \
--mode=pretrain_inpainter \
 \
 --batch_size=16 --inp_lr=1e-4 \
 \
 --summaries_secs=60 --ckpt_secs=10000 \