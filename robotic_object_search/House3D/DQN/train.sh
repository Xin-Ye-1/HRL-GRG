#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/train.py \
  --is_approaching_policy=False \
  --max_episodes=100000 \
  --max_episode_steps=1000 \
  --max_lowlevel_episode_steps=50 \
  --window_size=10 \
  --num_labels=78 \
  --a_size=6 \
  --history_steps=4 \
  --er=0.01 \
  --er=2000 \
  --er=0.01 \
  --epsilon=1 \
  --epsilon=10000 \
  --epsilon=0.1 \
  --use_gt=False \
  --curriculum_training=True \
  --highlevel_lr=0.0001 \
  --lowlevel_lr=0.0001 \
  --skip_frames=1 \
  --lowlevel_update_freq=10 \
  --highlevel_update_freq=100 \
  --target_update_freq=100000 \
  --batch_size=64 \
  --replay_start_size=0 \
  --vision_feature_pattern='_deeplab_depth_logits_10' \
  --depth_feature_pattern='_deeplab_depth_depth1_10' \
  --load_model=True \
  --continuing_training=False \
  --pretrained_model_path="../A3C/result_me_for_pretrain/model" \
  --model_path="${CURRENT_DIR}/result_me_pretrain/model" \
  --num_threads=1 \
  --num_scenes=4 \
  --num_targets=6 \
  --use_default_scenes=True \
  --use_default_targets=True \
  --default_scenes='5cf0e1e9493994e483e985c436b9d3bc' \
  --default_scenes='0c9a666391cc08db7d6ca1a926183a76' \
  --default_scenes='0c90efff2ab302c6f31add26cd698bea' \
  --default_scenes='00d9be7210856e638fa3b1addf2237d6' \
  --default_targets='sofa' \
  --default_targets='television' \
  --default_targets='tv_stand' \
  --default_targets='bed' \
  --default_targets='toilet' \
  --default_targets='bathtub' \
  --default_targets='music' \
  --default_targets='television' \
  --default_targets='table' \
  --default_targets='stand' \
  --default_targets='dressing_table' \
  --default_targets='heater' \


