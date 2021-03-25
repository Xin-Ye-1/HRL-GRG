#!/bin/bash



set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/train.py \
  --max_episodes=100000 \
  --max_episode_steps=100 \
  --max_lowlevel_episode_steps=10 \
  --window_size=7 \
  --num_channels=17 \
  --num_actions=4 \
  --num_envs=100 \
  --num_goals=16 \
  --env_dir='maps_16X16_v6' \
  --epsilon=1 \
  --epsilon=10000 \
  --epsilon=0.1 \
  --curriculum_training=True \
  --highlevel_lr=0.0001 \
  --lowlevel_lr=0.0001 \
  --skip_frames=1 \
  --history_steps=1 \
  --lowlevel_update_freq=10 \
  --highlevel_update_freq=10 \
  --target_update_freq=10000 \
  --batch_size=64 \
  --load_model=False \
  --continuing_training=False \
  --pretrained_model_path="${CURRENT_DIR}/result_for_pretrain/model" \
  --model_path="${CURRENT_DIR}/result/model" \
  --evaluate_file='../random_method/maps_16X16_v6_valid.txt' \
  --evaluate_during_training=True



