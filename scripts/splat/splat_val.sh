#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export WANDB_MODE=offline
export WANDB_MODE=disabled



# TRAIN

set CURR_DBSCAN 0.95
set CURR_TOPK 100
set CURR_QUERY 100

python main_instance_segmentation.py \
general.experiment_name="splat_train_5" \
general.project_name="splat_train" \
data/datasets=splat \
general.num_targets=2 \
data.num_labels=2 \
data.voxel_size=0.02 \
general.ignore_class_threshold=20 \
trainer.max_epochs=1001 \
trainer.check_val_every_n_epoch=10 \
general.save_visualizations=true \
general.eval_on_segments=false \
general.train_on_segments=false \
data.batch_size=4 \
model.num_queries=$CURR_QUERY \
general.export=true \
general.visualization_point_size=2


# TEST

set CURR_DBSCAN 0.95
set CURR_TOPK 84
set CURR_QUERY 150

python main_instance_segmentation.py \
general.experiment_name="validation_query_$CURR_QUERY_topk_$CURR_TOPK_dbscan_$CURR_DBSCAN" \
general.project_name="splat_eval" \
data/datasets=splat \
general.num_targets=2 \
data.num_labels=2 \
data.voxel_size=0.02 \
general.ignore_class_threshold=20 \
general.save_visualizations=true \
general.checkpoint='saved/splat_train_4/last.ckpt' \
general.train_mode=false \
general.eval_on_segments=false \
general.train_on_segments=false \
model.num_queries=$CURR_QUERY \
general.topk_per_image=$CURR_TOPK \
general.use_dbscan=true \
general.dbscan_eps=$CURR_DBSCAN \
general.export=true \
general.visualization_point_size=2
