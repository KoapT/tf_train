#!/bin/bash

source cfg.properties

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
WORK_DIR2="${CURRENT_DIR}/../../.."
EXE_DIR="${WORK_DIR}/../src"
PRETRAIN_MODEL_DIR="${WORK_DIR}/pretrain_models/${MODEL_NAME}"
SAMPLE_DIR="${WORK_DIR2}/samples_to_train/${SAMPLE_NAME}"
MODEL_CONFIG_PATH="${PRETRAIN_MODEL_DIR}/config.json"
LABEL_MAP_PATH="${SAMPLE_DIR}/LabelMap/label_map.json"
IMAGE_DIR="${SAMPLE_DIR}/JPEGImages/"
TRAIN_DIR="${SAMPLE_DIR}/train_dir/"
PB_PATH="${TRAIN_DIR}/saved_model/model.pb"

# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:${WORK_DIR2}/projects

if [ ! -f ${PB_PATH} ]; then
  echo 'Start export ------------'
  python ${EXE_DIR}/export.py \
   --checkpoint_path=${TRAIN_DIR} \
   --export_path=${PB_PATH} \
   --model_info_path=${MODEL_CONFIG_PATH} \
   --sample_info_path=${LABEL_MAP_PATH}
  echo "Export finished!"
fi

python ${EXE_DIR}/infer.py \
 --pb_path=${PB_PATH} \
 --images_dir=${IMAGE_DIR} \
 --model_info_path=${MODEL_CONFIG_PATH} \
 --sample_info_path=${LABEL_MAP_PATH} \
 --coco_id=${COCO_ID} \
 --batch_size=${INFER_BATCH_SIZE}


