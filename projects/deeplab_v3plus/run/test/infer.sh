#!/bin/bash



CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
WORK_DIR2="${CURRENT_DIR}/../../../.."
EXE_DIR="${WORK_DIR}/../../src"
IMGS_DIR="${WORK_DIR}/test_imgs"
CKPT_DIR="${WORK_DIR}/test_ckpt"
CKPT_PATH="${CKPT_DIR}/frozen_inference_graph.pb"
LABEL_MAP_PATH="${CKPT_DIR}/label_map.json"

# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:${WORK_DIR2}/models_master/research:${WORK_DIR2}/models_master/research/slim


python ${EXE_DIR}/infer.py \
  --checkpoint_path=${CKPT_PATH} \
  --images_dir=${IMGS_DIR} \
  --predictions_save_dir=${IMGS_DIR} \
  --predictions_color_save_dir=${IMGS_DIR} \
  --label_map_path=${LABEL_MAP_PATH}
