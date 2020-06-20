CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path ../data/multi_dsprites_data/multi_dsprites_colored_on_colored.tfrecords \
--ckpt_path 'path to checkpoint including segmentation network' \
--batch_size 8 --num_branch 5 --dataset_name multi_dsprites