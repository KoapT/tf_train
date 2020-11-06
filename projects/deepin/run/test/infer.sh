#!/bin/bash



CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
WORK_DIR2="${CURRENT_DIR}/../../../.."
EXE_DIR="${WORK_DIR}/../../src"
IMGS_DIR="${WORK_DIR}/test_imgs"
CKPT_PATH="${WORK_DIR}/test_ckpt/frozen_inference_graph.pb"
MODEL_CONFIG_PATH="${WORK_DIR}/test_ckpt/config.json"
LABEL_MAP_PATH="${WORK_DIR}/test_ckpt/label_map.json"

# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:${WORK_DIR2}/projects


python ${EXE_DIR}/infer.py \
  --images_dir=${IMGS_DIR} \
  --checkpoint_path=${CKPT_PATH} \
  --model_info_path=${MODEL_CONFIG_PATH} \
  --sample_info_path=${LABEL_MAP_PATH}
