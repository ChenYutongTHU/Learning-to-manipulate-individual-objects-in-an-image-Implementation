CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path ../data/objects_room_data/objects_room_train.tfrecords  \
--ckpt_path 'path to checkpoint including segmentation network' \
--batch_size 8 --num_branch 6 --dataset_name objects_room
