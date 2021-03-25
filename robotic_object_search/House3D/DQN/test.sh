#!/bin/bash



set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/test.py \
  --is_approaching_policy=False \
  --max_episodes=1 \
  --max_episode_steps=1000 \
  --max_lowlevel_episode_steps=50 \
  --window_size=10 \
  --num_labels=78 \
  --a_size=6 \
  --history_steps=4 \
  --er=0.5 \
  --er=10000 \
  --er=0.01 \
  --use_gt=False \
  --lowlevel_lr=0.0001 \
  --vision_feature_pattern='_deeplab_depth_logits_10' \
  --depth_feature_pattern='_deeplab_depth_depth1_10' \
  --load_model=True \
  --model_path="${CURRENT_DIR}/result_se_pretrain/model" \
  --num_scenes=1 \
  --num_targets=1 \
  --use_default_scenes=True \
  --use_default_targets=True \
  --default_scenes='5cf0e1e9493994e483e985c436b9d3bc' \
  --default_targets='bed' \
  --default_targets='mirror' \
  --default_targets='ottoman' \
  --default_targets='sofa' \
  --default_targets='desk' \
  --default_targets='picture_frame' \
#  --default_targets='music' \
#  --default_targets='television' \
#  --default_targets='table' \
#  --default_targets='stand' \
#  --default_targets='dressing_table' \
#  --default_targets='heater' \