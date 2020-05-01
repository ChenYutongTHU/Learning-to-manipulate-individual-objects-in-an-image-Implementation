CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
--data_path /home/yutong/Learning-to-Manipulate-Individual-Objects-in-an-Image/data/flying_animals_data/img_data.npz \
--ckpt_path /media/DATA2_6TB/yutong/learning2manip/fa/end2end/15_genlr1e-5_w1e-6_inplr1e-4/deaac_texdim20_beta4_bgdim30_beta4_maskdim20_gamma500_5e-5_bs4_lr1e-4/save/model-0 \
--batch_size 8 --dataset_name flying_animals 


#fa_ckpt/model-110000