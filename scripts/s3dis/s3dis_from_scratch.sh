#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_AREA=1  # set the area number accordingly [1,6]
CURR_DBSCAN=0.6
CURR_TOPK=-1
CURR_QUERY=100

set CURR_AREA 1
set CURR_DBSCAN 0.6
set CURR_TOPK -1
set CURR_QUERY 100

python main_instance_segmentation.py \
  general.project_name="s3dis" \
  general.experiment_name="area${CURR_AREA}_from_scratch" \
  data.batch_size=4 \
  data/datasets=s3dis \
  general.num_targets=14 \
  data.num_labels=13 \
  trainer.max_epochs=1001 \
  general.area=$CURR_AREA \
  trainer.check_val_every_n_epoch=10

python main_instance_segmentation.py \
  general.project_name="s3dis" \
  general.experiment_name="area1_from_scratch" \
  data.batch_size=4 \
  data/datasets=s3dis \
  general.num_targets=14 \
  data.num_labels=13 \
  trainer.max_epochs=1001 \
  general.area=1 \
  trainer.check_val_every_n_epoch=10


python main_instance_segmentation.py \
general.project_name="s3dis_eval" \
general.experiment_name="area${CURR_AREA}_from_scratch_eps_${CURR_DBSCAN}_topk_${CURR_TOPK}_q_${CURR_QUERY}" \
general.checkpoint="checkpoints/s3dis/from_scratch/area${CURR_AREA}.ckpt" \
general.train_mode=false \
data.batch_size=4 \
data/datasets=s3dis \
general.num_targets=14 \
data.num_labels=13 \
general.area=${CURR_AREA} \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}
