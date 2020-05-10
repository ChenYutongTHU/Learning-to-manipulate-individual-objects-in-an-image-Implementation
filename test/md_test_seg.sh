CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path /home/yutong/Learning-to-Manipulate-Individual-Objects-in-an-Image/data/multi_dsprites_data/multi_dsprites_colored_on_colored.tfrecords \
--ckpt_path ../save_checkpoint/md/model-0 \
--batch_size 8 --num_branch 5 --dataset_name multi_dsprites