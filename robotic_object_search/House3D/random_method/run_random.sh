#!/bin/bash



set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/random_walk.py \
  --evaluate_file='1s6t.txt' \




