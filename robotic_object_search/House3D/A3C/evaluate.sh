#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/evaluate.py \
  --model_path="result_se_pretrain/model" \
  --evaluate_file='../random_method/1s6t.txt'


