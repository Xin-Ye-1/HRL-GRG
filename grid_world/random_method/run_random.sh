#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/random_walk.py \
  --max_episodes=100 \
  --max_episode_steps=100 \
  --num_actions=4 \
  --num_envs=20 \
  --num_goals=16 \
  --env_dir='maps_16X16_v6' \
  --evaluate_file='maps_16X16_v6_valid_seengoals.txt' \
  --save_path='' \



