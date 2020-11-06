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
FINE_TUNE_CHECKPOINT_PATH="${PRETRAIN_MODEL_DIR}/model.ckpt"
MODEL_CONFIG_PATH="${PRETRAIN_MODEL_DIR}/config.json"
LABEL_MAP_PATH="${SAMPLE_DIR}/LabelMap/label_map.json"
TRAIN_SET_PATH="${SAMPLE_DIR}/Set/train.txt"
EVAL_SET_PATH="${SAMPLE_DIR}/Set/eval.txt"
TRAIN_TF_RECORD_PATH="${SAMPLE_DIR}/TFRecord/train*.record"
EVAL_TF_RECORD_PATH="${SAMPLE_DIR}/TFRecord/eval*.record"
COCO_ANN_PATH="${SAMPLE_DIR}/cocoAnno/annotations_eval.json"
IMAGE_DIR="${SAMPLE_DIR}/JPEGImages/"
ANN_DIR="${SAMPLE_DIR}/Annotations/"
TRAIN_DIR="${SAMPLE_DIR}/train_dir/"
PB_PATH="${TRAIN_DIR}/saved_model/model.pb"
EVAL_DIR="${TRAIN_DIR}/eval_log/"

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
echo COCO annotation file path:$COCO_ANN_PATH
# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:${WORK_DIR2}/projects


smax=0
rest_cnt=0
while [ $smax -lt $NUM_STEPS ]
do
	echo "Train run--------------------"
	if [ ${smax} != 0 ]; then
	  INITIALIZE_OUTPUT_LAYERS=true
	fi
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
	echo "sleep several seconds-------------"
	sleep ${ONCE_REST_SECS}s
	echo "Train run done,export start-------------"

	python ${EXE_DIR}/export.py \
	 --checkpoint_path=${TRAIN_DIR} \
	 --export_path=${PB_PATH} \
	 --model_info_path=${MODEL_CONFIG_PATH} \
	 --sample_info_path=${LABEL_MAP_PATH}
	echo "Export finished!"

	echo "sleep several seconds-------------"
	sleep ${ONCE_REST_SECS}s

	rest_cnt=`expr $rest_cnt + 1`
	# If need to evaluate once
	if [ $EVAL_INTERVAL_RESTS -gt 0 ]; then
		if [ $rest_cnt -ge $EVAL_INTERVAL_RESTS ]; then
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
			rest_cnt=0
		fi
	fi
	echo "sleep several seconds-------------"
	sleep ${ONCE_REST_SECS}s

	# Find last check point
  str=`grep "model_checkpoint_path: *" ${TRAIN_DIR}/checkpoint`
  name=`echo $str | grep -P 'model.ckpt-\d+' -o`
  smax=`echo ${name##*-}`
  echo step $smax has finished train/export/evaluation
  echo "====================================================================="
done