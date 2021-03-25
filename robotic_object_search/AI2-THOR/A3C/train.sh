#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/train.py \
  --is_approaching_policy=False \
  --max_episodes=100000 \
  --max_lowlevel_episode_steps=10 \
  --window_size=30 \
  --history_steps=4 \
  --curriculum_training=True \
  --highlevel_lr=0.0001 \
  --lowlevel_lr=0.0001 \
  --skip_frames=1 \
  --lowlevel_update_freq=10 \
  --highlevel_update_freq=100 \
  --target_update_freq=100000 \
  --batch_size=64 \
  --replay_start_size=0 \
  --load_model=True \
  --continuing_training=False \
  --pretrained_model_path="result_for_pretrain/model" \
  --model_path="${CURRENT_DIR}/result_pretrain/model" \
  --num_threads=1 \
  --num_train_scenes=20 \
  --num_validate_scenes=5 \
  --num_test_scenes=5 \
  --min_step_threshold=0 \
  --is_training=True \


