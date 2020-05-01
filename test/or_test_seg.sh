CUDA_VISIBLE_DEVICES=1 python test_segmentation.py \
--data_path ../data/objects_room_data/objects_room_train.tfrecords  \
--ckpt_path or_ckpt/model-176000 \
--batch_size 8 --dataset_name objects_room