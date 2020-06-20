ABSPATH=$(readlink -f $0)


CUDA_VISIBLE_DEVICES=0 python main.py  \
--checkpoint_dir=checkpoint/flying_animals/pretrain_inpainter \
--sh_path=$ABSPATH \
 \
--dataset=flying_animals  --root_dir=data/flying_animals_data/img_data.npz \
--mode=pretrain_inpainter  --max_num=5 \
 \
 --batch_size=4 --inp_lr=1e-4 \
 \
 --summaries_secs=180 --ckpt_secs=5000 \