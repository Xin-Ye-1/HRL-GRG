#!/bin/bash


set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/test.py \
  --max_episodes=1 \
  --max_episode_steps=100 \
  --max_lowlevel_episode_steps=10 \
  --window_size=7 \
  --highlevel_num_channels=17 \
  --lowlevel_num_channels=2 \
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
  --lowlevel_update_freq=10 \
  --highlevel_update_freq=10 \
  --target_update_freq=100000 \
  --batch_size=64 \
  --load_model=True \
  --continuing_training=False \
  --pretrained_model_path="" \
  --model_path="${CURRENT_DIR}/result_pretrain/model" \
  --evaluate_file='../random_method/maps_16X16_valid_v6.txt'