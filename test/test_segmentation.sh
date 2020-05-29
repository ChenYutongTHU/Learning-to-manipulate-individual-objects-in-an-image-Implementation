CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path ../data/flying_animals_data/img_data.npz \
--ckpt_path 'path to checkpoint including segmentation network' \
--batch_size 8 --num_branch 6 --dataset_name flying_animals 

CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path ../data/multi_dsprites_data/multi_dsprites_colored_on_colored.tfrecords \
--ckpt_path 'path to checkpoint including segmentation network' \
--batch_size 8 --num_branch 5 --dataset_name multi_dsprites

CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path ../data/objects_room_data/objects_room_train.tfrecords  \
--ckpt_path 'path to checkpoint including segmentation network' \
--batch_size 8 --num_branch 6 --dataset_name objects_room

CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path ../data/multi_texture_data \
--ckpt_path 'path to checkpoint including segmentation network' \
--batch_size 8 --num_branch 5 --dataset_name multi_texture 