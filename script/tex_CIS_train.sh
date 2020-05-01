CURDIR=`/bin/pwd`
BASEDIR=$(dirname $0)
ABSPATH=$(readlink -f $0)
ABSDIR=$(dirname $ABSPATH)


echo "CURDIR is $CURDIR"
echo "BASEDIR is $BASEDIR"
echo "ABSPATH is $ABSPATH"
echo "ABSDIR is $ABSDIR"

CUDA_VISIBLE_DEVICES=0 python main.py  --sh_path=$ABSPATH  \
--checkpoint_dir=/media/DATA2_6TB/yutong/learning2manip_inp/multi_texture/CIS/repeat_useold_inp \
--dataset=multi_texture --max_num=4 --num_branch=5 --root_dir=data/multi_texture_data/ \
--mode=train_CIS --model=resnet_v2_50 \
\
\
--resume_inpainter=True --resume_fullmodel=False \
--fullmodel_ckpt=/  \
--inpainter_ckpt=/media/DATA2_6TB/yutong/learning2manip/multi_texture/CPN_base/pretrain_inpainter/09e67_bs16_lr1e-4/Inpainter_Sum/Inpainter-73333 \
\
--batch_size=8 --inp_lr=1e-4 --gen_lr=3e-5 \
--gen_clip_value=-1 \
--epsilon=30 --iters_inp=1 --iters_gen=3 \
--ckpt_steps=2000 \