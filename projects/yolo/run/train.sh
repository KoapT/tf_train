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
echo "log every n steps: ${LOG_STEPS}"
echo "save summaries interval seconds: ${SAVE_SUMMARIES_SECS}"
echo "save checkpoint interval seconds: ${SAVE_CKPT_INTERVAL_SECS}"
echo "rest interval seconds: ${REST_INTERVAL_SECS}"
echo "once rest seconds: ${ONCE_REST_SECS}"
echo "eval interval rests: ${EVAL_INTERVAL_RESTS}"
echo "======================================"




# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
WORK_DIR2="${CURRENT_DIR}/../../.."
EXE_DIR="${WORK_DIR}/../src"
PRETRAIN_MODEL_DIR="${WORK_DIR}/pretrain_models/${MODEL_NAME}"
SAMPLE_DIR="${WORK_DIR2}/samples_to_train/${SAMPLE_NAME}"
FINE_TUNE_CHECKPOINT_PATH="${PRETRAIN_MODEL_DIR}/model_544_eval.ckpt"
MODEL_CONFIG_PATH="${PRETRAIN_MODEL_DIR}/config.json"
LABEL_MAP_PATH="${SAMPLE_DIR}/LabelMap/label_map.json"
TRAIN_SET_PATH="${SAMPLE_DIR}/Set/train.txt"
TRAIN_TF_RECORD_PATH="${SAMPLE_DIR}/TFRecord/train*.record"
TRAIN_DIR="${SAMPLE_DIR}/train_dir/"

# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:${WORK_DIR2}/projects



echo "Train run--------------------"
python ${EXE_DIR}/train.py \
  --training_number_of_steps=${NUM_STEPS} \
  --num_clones=${NUM_HARDWARE_CORES} \
  --initialize_last_layer_from_checkpoint=${INITIALIZE_OUTPUT_LAYERS} \
  --logdir=${TRAIN_DIR} \
  --tf_initial_checkpoint_path=${FINE_TUNE_CHECKPOINT_PATH} \
  --model_info_path=${MODEL_CONFIG_PATH} \
  --sample_info_path=${LABEL_MAP_PATH} \
  --sample_path=${TRAIN_TF_RECORD_PATH} \
  --sample_list_path=${TRAIN_SET_PATH} \
  --log_steps=${LOG_STEPS} \
  --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
  --save_interval_secs=${SAVE_CKPT_INTERVAL_SECS} \
  --rest_interval_secs=${REST_INTERVAL_SECS} \
  --once_rest_secs=${ONCE_REST_SECS}
echo "Train run done"

