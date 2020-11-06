#!/bin/bash


# Exit immediately if a command exits with a non-zero status.
#set -e

source cfg.properties
echo "============begin to train============"
echo "model name: ${MODEL_NAME}"
echo "sample name: ${SAMPLE_NAME}"
echo "number of steps: ${NUM_STEPS}" 
echo "number of hardware cores: ${NUM_HARDWARE_CORES}"
echo "initialize output layers: ${INITIALIZE_OUTPUT_LAYERS}"
echo "======================================"




# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
WORK_DIR2="${CURRENT_DIR}/../../.."
EXE_DIR="${WORK_DIR}/../src"
PRETRAIN_MODEL_DIR="${WORK_DIR}/pretrain_models/${MODEL_NAME}"
SAMPLE_DIR="${WORK_DIR2}/samples_to_train/${SAMPLE_NAME}"
FINE_TUNE_CHECKPOINT_PATH="${PRETRAIN_MODEL_DIR}/model.ckpt"
CONFIG_PATH="${PRETRAIN_MODEL_DIR}/pipeline.json"
LABEL_MAP_PATH="${SAMPLE_DIR}/LabelMap/label_map.json"
TRAIN_SET_PATH="${SAMPLE_DIR}/Set/train.txt"
EVAL_SET_PATH="${SAMPLE_DIR}/Set/eval.txt"
TRAIN_TF_RECORD_PATH="${SAMPLE_DIR}/TFRecord/train.record"
EVAL_TF_RECORD_PATH="${SAMPLE_DIR}/TFRecord/eval.record"
TRAIN_DIR="${SAMPLE_DIR}/train_dir"

# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:${WORK_DIR2}/models_master/research:${WORK_DIR2}/models_master/research/slim

echo "train run--------------------"
python ${EXE_DIR}/train.py \
  --logtostderr \
  --training_number_of_steps=${NUM_STEPS} \
  --num_clones=${NUM_HARDWARE_CORES} \
  --initialize_last_layer=${INITIALIZE_OUTPUT_LAYERS} \
  --tf_initial_checkpoint=${FINE_TUNE_CHECKPOINT_PATH} \
  --train_logdir=${TRAIN_DIR} \
  --label_map_path=${LABEL_MAP_PATH} \
  --tfrecord_path=${TRAIN_TF_RECORD_PATH} \
  --list_path=${TRAIN_SET_PATH} \
  --config_path=${CONFIG_PATH}
echo "train run done"

# Run evaluation. This performs eval over the full val split and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=75.34%.
echo "evaluation run-----------------------"
python ${EXE_DIR}/eval.py \
  --logtostderr \
  --checkpoint_dir=${TRAIN_DIR} \
  --eval_logdir=${TRAIN_DIR} \
  --label_map_path=${LABEL_MAP_PATH} \
  --tfrecord_path=${EVAL_TF_RECORD_PATH} \
  --list_path=${EVAL_SET_PATH} \
  --config_path=${CONFIG_PATH}
echo "evaluation run done"

# Visualize the results.
echo "visualization run------------------"
python ${EXE_DIR}/vis.py \
  --logtostderr \
  --checkpoint_dir=${TRAIN_DIR} \
  --vis_logdir=${TRAIN_DIR} \
  --label_map_path=${LABEL_MAP_PATH} \
  --tfrecord_path=${EVAL_TF_RECORD_PATH} \
  --list_path=${EVAL_SET_PATH} \
  --config_path=${CONFIG_PATH}
echo "visualization run done"
