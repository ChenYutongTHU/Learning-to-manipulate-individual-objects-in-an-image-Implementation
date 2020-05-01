CURDIR=`/bin/pwd`
BASEDIR=$(dirname $0)
ABSPATH=$(readlink -f $0)
ABSDIR=$(dirname $ABSPATH)


echo "CURDIR is $CURDIR"
echo "BASEDIR is $BASEDIR"
echo "ABSPATH is $ABSPATH"
echo "ABSDIR is $ABSDIR"

CUDA_VISIBLE_DEVICES=0 python main.py  \
--checkpoint_dir=/media/DATA2_6TB/yutong/learning2manip_inp/multi_texture/pretrain_inpainter/bs16_lr1e-4 \
--sh_path=$ABSPATH \
\
--dataset=multi_texture  --max_num=4 --root_dir=data/multi_texture_data \
--mode=pretrain_inpainter \
 \
 --batch_size=16 --inp_lr=1e-4 \
 \
 --summaries_secs=30 --ckpt_secs=200 \
