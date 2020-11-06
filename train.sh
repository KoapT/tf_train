#!/bin/bash
# From the models_maste/research/object_detection/ directory


source cfg.properties
echo "============begin to train============"
echo "model name: ${MODEL_NAME}"
echo "sample name: ${SAMPLE_NAME}"
echo "number of steps: ${NUM_STEPS}" 
echo "======================================"


# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`/models_master/research:`pwd`/models_master/research/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
EXE_DIR="${WORK_DIR}/models_master/research/object_detection"
PRETRAIN_MODEL_DIR="${WORK_DIR}/pretrain_models/${MODEL_NAME}"
SAMPLE_DIR="${WORK_DIR}/samples_to_train/${SAMPLE_NAME}"
PIPELINE_CONFIG_PATH="${PRETRAIN_MODEL_DIR}/pipeline.config"
FINE_TUNE_CHECKPOINT_PATH="${PRETRAIN_MODEL_DIR}/model.ckpt"
TRAIN_DIR="${SAMPLE_DIR}/train_dir"
LABEL_MAP_PATH="${SAMPLE_DIR}/LabelMap/label_map.pbtxt"
TRAIN_TF_RECORD_PATH="${SAMPLE_DIR}/TFRecord/train.record"
EVAL_TF_RECORD_PATH="${SAMPLE_DIR}/TFRecord/eval.record"


echo "synchronize pipeline.config file: ${PIPELINE_CONFIG_PATH}------------------"
python ${WORK_DIR}/modify_pipeline_config.py \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	--num_steps=${NUM_STEPS} \
	--fine_tune_checkpoint=${FINE_TUNE_CHECKPOINT_PATH} \
	--train_label_map_path=${LABEL_MAP_PATH} \
	--train_tf_record_path=${TRAIN_TF_RECORD_PATH} \
	--eval_label_map_path=${LABEL_MAP_PATH} \
	--eval_tf_record_path=${EVAL_TF_RECORD_PATH}
echo "synchronize done"

echo "train run--------------------------------"
python ${EXE_DIR}/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${TRAIN_DIR} \
    --alsologtostderr
echo "train run done"

