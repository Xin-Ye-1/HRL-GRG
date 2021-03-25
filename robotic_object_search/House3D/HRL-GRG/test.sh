#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/test.py \
  --is_approaching_policy=False \
  --max_episodes=100 \
  --max_episode_steps=1000 \
  --max_lowlevel_episode_steps=50 \
  --window_size=10 \
  --num_labels=78 \
  --a_size=6 \
  --history_steps=4 \
  --use_gt=False \
  --lowlevel_lr=0.0001 \
  --vision_feature_pattern='_deeplab_depth_logits_10' \
  --depth_feature_pattern='_deeplab_depth_depth1_10' \
  --load_model=True \
  --model_path="${CURRENT_DIR}/result_se_pretrain/model" \
  --evaluate_file='../random_method/1s6t.txt' \
  --num_scenes=1 \
  --num_targets=1 \
  --use_default_scenes=True \
  --use_default_targets=True \
  --default_scenes='0880799c157b4dff08f90db221d7f884' \
  --default_targets='television' \
#  --default_scenes='07d1d46444ca33d50fbcb5dc12d7c103' \
#  --default_scenes='026c1bca121239a15581f32eb27f2078' \
#  --default_scenes='0147a1cce83b6089e395038bb57673e3' \
#  --default_scenes='0880799c157b4dff08f90db221d7f884' \
#  --default_scenes='5cf0e1e9493994e483e985c436b9d3bc' \
#  --default_targets='ottoman' \
#  --default_targets='television' \
#  --default_targets='table' \
#  --default_targets='stand' \
#  --default_targets='dressing_table' \
#  --default_targets='heater'