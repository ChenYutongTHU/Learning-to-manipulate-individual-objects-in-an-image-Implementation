CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path ../data/flying_animals_data/img_data.npz \
--ckpt_path 'path to checkpoint including segmentation network' \
--batch_size 8 --num_branch 6 --dataset_name flying_animals 
