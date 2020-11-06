#!/bin/bash
# From the models_maste/research/object_detection/ directory


source cfg.properties


# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`/models_master/research:`pwd`/models_master/research/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
EXE_DIR="${WORK_DIR}/models_master/research/object_detection"
EXPORT_DIR="${WORK_DIR}/export_dir/${MODEL_NAME}"
EXPORT_SUB_DIR="${EXPORT_DIR}/`date +%Y%m%d%H%M%S`_${SAMPLE_NAME}"
EXPORT_PATH="${EXPORT_SUB_DIR}/frozen_inference_graph.pb"
PRETRAIN_MODEL_DIR="${WORK_DIR}/pretrain_models/${MODEL_NAME}"
SAMPLE_DIR="${WORK_DIR}/samples_to_train/${SAMPLE_NAME}"
TRAIN_DIR="${SAMPLE_DIR}/train_dir"
LABEL_MAP_PATH="${SAMPLE_DIR}/LabelMap/label_map.pbtxt"
PIPELINE_CONFIG_PATH="${TRAIN_DIR}/pipeline.config"


INPUT_TYPE=image_tensor


echo "Start to export===================="
echo "pipeline.config path: ${PIPELINE_CONFIG_PATH}"
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

TRAINED_CKPT_PREFIX="${TRAIN_DIR}/model.ckpt-${smax}"
echo "trained ckpt directory: ${TRAIN_DIR}"
echo "trained ckpt prefix: ${TRAINED_CKPT_PREFIX}  ------------------"

#cp ${TRAINED_CKPT_PREFIX}.data-00000-of-00001 ${EXPORT_SUB_DIR}/model.ckpt.data-00000-of-00001
#cp ${TRAINED_CKPT_PREFIX}.index ${EXPORT_SUB_DIR}/model.ckpt.index
#cp ${TRAINED_CKPT_PREFIX}.meta ${EXPORT_SUB_DIR}/model.ckpt.meta
#cp ${TRAIN_DIR}/checkpoint ${EXPORT_SUB_DIR}/checkpoint
#cp ${TRAIN_DIR}/pipeline.config ${EXPORT_SUB_DIR}/pipeline.config
echo "run export python file ----------------------------------"
python ${EXE_DIR}/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_SUB_DIR}
echo "run export python file done"

if [ ! -f "$EXPORT_PATH" ]; then
	echo "export failed!"
	rm -rf ${EXPORT_SUB_DIR}
else
	echo "export success"
	echo "copy label map of sample to export directory"
	cp ${LABEL_MAP_PATH} ${EXPORT_SUB_DIR}/

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


