#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/random_walk.py \
  --is_approaching_policy=False \
  --max_episodes=250 \
  --window_size=30 \
  --num_train_scenes=20 \
  --num_validate_scenes=5 \
  --num_test_scenes=5 \
  --min_step_threshold=5 \
  --save_path='' \
  --evaluate_file='ssso.txt' \
  --seen_scenes=False \
  --seen_objects=False \




