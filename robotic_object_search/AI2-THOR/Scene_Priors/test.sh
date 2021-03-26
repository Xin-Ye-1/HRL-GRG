#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/test.py \
  --is_approaching_policy=False \
  --max_episodes=1 \
  --vision_size=2048 \
  --word_size=100 \
  --score_size=1000 \
  --history_steps=4 \
  --curriculum_training=False \
  --lowlevel_lr=0.0001 \
  --lowlevel_update_freq=30 \
  --batch_size=64 \
  --replay_start_size=0 \
  --load_model=False \
  --continuing_training=False \
  --model_path="${CURRENT_DIR}/result/model" \
  --num_threads=1 \
  --num_train_scenes=20 \
  --num_validate_scenes=5 \
  --num_test_scenes=5 \
  --min_step_threshold=0 \
  --is_training=True \
  --wemb_path="gcn/glove_map100d.hdf5" \
