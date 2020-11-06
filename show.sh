#!/bin/bash
# launch tensorboard
# then you can open the browser with "http://localhost:6006" to display train information

source cfg.properties

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`/models_master/research:`pwd`/models_master/research/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
SHOW_DIR="${WORK_DIR}/samples_to_train/${SAMPLE_NAME}/train_dir"
USE_METHOD_DESCRIPTION="Open browser and input http://localhost:6006 to see!"

echo "Display train information from directory: ${SHOW_DIR}"
echo ${USE_METHOD_DESCRIPTION}
tensorboard --logdir=${SHOW_DIR}

