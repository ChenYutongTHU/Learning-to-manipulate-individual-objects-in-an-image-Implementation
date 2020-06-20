CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path data/multi_texture_data \
--ckpt_path 'path to checkpoint including segmentation network' \
--batch_size 8 --num_branch 3 --dataset_name multi_texture --PC=True --model=segnet
