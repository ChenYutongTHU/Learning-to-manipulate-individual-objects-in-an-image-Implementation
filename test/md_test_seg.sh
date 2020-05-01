CUDA_VISIBLE_DEVICES=1 python test_segmentation.py \
--data_path ../data/multi_dsprites_data/multi_dsprites_colored_on_colored.tfrecords \
--ckpt_path md_ckpt/model-100000 \
--batch_size 8 --dataset_name multi_dsprites