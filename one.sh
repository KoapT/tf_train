#!/bin/bash
# Run all programs in one time

source cfg.properties

CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
SAMPLE_DIR="${WORK_DIR}/samples_to_train/${SAMPLE_NAME}"


cd ${SAMPLE_DIR}
. ./loadData_createTfrecord.sh
echo " "
echo " "
echo " "
echo " "
echo " "
cd ${SAMPLE_DIR}/../..
. ./train_export.sh
