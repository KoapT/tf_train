#!/bin/bash

source cfg.properties


CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
WORK_DIR2="${CURRENT_DIR}/../../.."
EXE_DIR="${WORK_DIR}/../src"
PRETRAIN_MODEL_DIR="${WORK_DIR}/pretrain_models/${MODEL_NAME}"
SAMPLE_DIR="${WORK_DIR2}/samples_to_train/${SAMPLE_NAME}"
MODEL_CONFIG_PATH="${PRETRAIN_MODEL_DIR}/config.json"
LABEL_MAP_PATH="${SAMPLE_DIR}/LabelMap/label_map.json"
EVAL_SET_PATH="${SAMPLE_DIR}/Set/eval.txt"
COCO_ANN_PATH="${SAMPLE_DIR}/cocoAnno/annotations_eval.json"
IMAGE_DIR="${SAMPLE_DIR}/JPEGImages/"
ANN_DIR="${SAMPLE_DIR}/Annotations/"
TRAIN_DIR="${SAMPLE_DIR}/train_dir/"
PB_PATH="${TRAIN_DIR}/saved_model/model.pb"
EVAL_DIR="${TRAIN_DIR}/eval_log/"

# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:${WORK_DIR2}/projects

# Generate coco format annotation file
if [ ! -f ${COCO_ANN_PATH} ]; then
  echo 'Generate coco format annotation file------------'
  python ${SAMPLE_DIR}/gen_coco_anno.py \
  --anno_dir=${ANN_DIR} \
  --set_file=${EVAL_SET_PATH} \
  --label_map_path=${LABEL_MAP_PATH} \
  --out_coco_anno_path=${COCO_ANN_PATH}
  echo 'Finish!'
fi

if [ ! -f ${PB_PATH} ]; then
  echo 'Generate coco format annotation file------------'
  python ${EXE_DIR}/export.py \
   --checkpoint_path=${TRAIN_DIR} \
   --export_path=${PB_PATH} \
   --model_info_path=${MODEL_CONFIG_PATH} \
   --sample_info_path=${LABEL_MAP_PATH}
  echo "Export finished!"
fi

echo "Evaluation run-----------------------"
start_time=`date +%s%N`
python ${EXE_DIR}/evaluation.py \
  --export_path=${PB_PATH} \
  --ckpt_dir=${TRAIN_DIR} \
  --log_dir=${EVAL_DIR} \
  --model_info_path=${MODEL_CONFIG_PATH} \
  --coco_ann_path=${COCO_ANN_PATH} \
  --image_dir=${IMAGE_DIR} \
  --coco_id=${COCO_ID}

end_time=`date +%s%N`
use_time=`echo $end_time $start_time | awk '{print ($1 - $2) / 1000000000}'`
echo "Evaluation take time: ${use_time}s"
echo "Evaluation run done"





