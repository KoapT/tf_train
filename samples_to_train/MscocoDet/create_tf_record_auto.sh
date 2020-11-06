#!/bin/bash
# create samples record automatically

source sample_cfg.properties


CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
TRAIN_DIR="${WORK_DIR}/train_dir"



echo "Start to create samples record automatically"
echo "Samples to train in all samples occupy: ${TRAIN_RATIO}"
echo "------------------------"
echo "clear train directory: ${TRAIN_DIR}"
rm -rf ${TRAIN_DIR}/*
echo "    "
echo "    "
python writeFileName_to_setTxt.py --r ${TRAIN_RATIO}
echo "    "
echo "    "
sleep 1s
python create_tf_record.py
