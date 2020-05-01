CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path ../data/multi_texture_data \
--ckpt_path /media/DATA2_6TB/yutong/learning2manip_inp/multi_texture/CIS/repeat_useold_inp/save/model-158000 \
--batch_size 8 --num_branch 5 --dataset_name multi_texture 
