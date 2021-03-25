#!/bin/bash

set -e

CURRENT_DIR=$(pwd)

CUDA_VISIBLE_DEVICES=-1 python "${CURRENT_DIR}"/evaluate.py \
  --model_path="${CURRENT_DIR}/result_pretrain/model" \
  --evaluate_file='../random_method/ssso.txt'








