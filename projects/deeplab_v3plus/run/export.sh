#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
#set -e

source cfg.properties
echo "============begin to export============"
echo "model name: ${MODEL_NAME}"
echo "sample name: ${SAMPLE_NAME}"
echo "number of steps: ${NUM_STEPS}"
echo "======================================="



# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
WORK_DIR2="${CURRENT_DIR}/../../.."
EXE_DIR="${WORK_DIR}/../src"
EXPORT_DIR="${WORK_DIR}/export_dir/${MODEL_NAME}"
EXPORT_SUB_DIR="${EXPORT_DIR}/`date +%Y%m%d%H%M%S`_${SAMPLE_NAME}"
PRETRAIN_MODEL_DIR="${WORK_DIR}/pretrain_models/${MODEL_NAME}"
SAMPLE_DIR="${WORK_DIR2}/samples_to_train/${SAMPLE_NAME}"
TRAIN_DIR="${SAMPLE_DIR}/train_dir"
CONFIG_PATH="${PRETRAIN_MODEL_DIR}/pipeline.json"
LABEL_MAP_PATH="${SAMPLE_DIR}/LabelMap/label_map.json"


# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:${WORK_DIR2}/models_master/research:${WORK_DIR2}/models_master/research/slim


echo "Start to export===================="
echo "configure file path: ${CONFIG_PATH}"
echo "export directory: ${EXPORT_SUB_DIR}"
# Create save directory
if [ ! -d ${EXPORT_DIR} ]; then
	mkdir ${EXPORT_DIR}
fi
if [ ! -d ${EXPORT_SUB_DIR} ]; then
	mkdir ${EXPORT_SUB_DIR}
fi
# Find last check point
cd ${TRAIN_DIR}
list_str=`ls model.ckpt-*.meta`
OLD_IFS="$IFS" 
IFS=" " 
list=($list_str) 
IFS="$OLD_IFS" 
smax=0
for s in ${list[@]} 
do 
	s=`echo ${s} | awk -F. '{print $2}'` 
	s=`echo ${s} | awk -F- '{print $2}'`    
	if [ ${s} -gt ${smax} ]; then
		smax=$s
	fi
done
cd ${WORK_DIR}

# Export the trained checkpoint.
TRAINED_CKPT_PATH="${TRAIN_DIR}/model.ckpt-${smax}"
EXPORT_PATH="${EXPORT_SUB_DIR}/frozen_inference_graph.pb"
echo "trained ckpt path: ${TRAINED_CKPT_PATH}"
echo "export path:${EXPORT_PATH}"

#num_classes=$(python -c 'import utils; print(utils.getClassesNum("'${LABEL_MAP_PATH}'"))')
#echo "num_classes=${num_classes}"
echo "run export ----------------------------------"
python ${EXE_DIR}/export_model.py \
  --logtostderr \
  --label_map_path=${LABEL_MAP_PATH} \
  --checkpoint_path=${TRAINED_CKPT_PATH} \
  --export_path=${EXPORT_PATH} \
  --config_path=${CONFIG_PATH}
echo "run export done"

if [ ! -f "$EXPORT_PATH" ]; then
	echo "export failed!"
	rm -rf ${EXPORT_SUB_DIR}
else
	echo "export success"
	echo "copy some files to export directory"
	cp ${CONFIG_PATH} ${EXPORT_SUB_DIR}/
	cp ${LABEL_MAP_PATH} ${EXPORT_SUB_DIR}/
	cp ${TRAIN_DIR}/graph.pbtxt ${EXPORT_SUB_DIR}/
	cp ${TRAIN_DIR}/checkpoint ${EXPORT_SUB_DIR}/
	cp ${TRAINED_CKPT_PATH}.data-00000-of-00001 ${EXPORT_SUB_DIR}/model.ckpt.data-00000-of-00001
	cp ${TRAINED_CKPT_PATH}.index ${EXPORT_SUB_DIR}/model.ckpt.index
	cp ${TRAINED_CKPT_PATH}.meta ${EXPORT_SUB_DIR}/model.ckpt.meta
	python -c 'import utils; print(utils.unifyCheckPointTxt("'${EXPORT_SUB_DIR}/checkpoint'"))'

	echo "clear and update--------------------------------"
	echo "remove temporary files for training in directory: ${TRAIN_DIR}"
	rm -rf ${TRAIN_DIR}/*
	echo "remove old pretrain model files for update in directory: ${PRETRAIN_MODEL_DIR}"
	rm -rf ${PRETRAIN_MODEL_DIR}/*
	echo "copy export files in directory: ${EXPORT_SUB_DIR}"
	echo "to pretrain model directory: ${PRETRAIN_MODEL_DIR}"
	cp -rf ${EXPORT_SUB_DIR}/* ${PRETRAIN_MODEL_DIR}/
	echo "clear and update done"
if


