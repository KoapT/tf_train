#!/bin/bash
# create samples record automatically



# user modify=======================================================
#PRE_CLR=true    #if clear Annotations and JPEGImages directories, true: do, false: not
# user modify end===================================================
source sample_cfg.properties



# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
IMG_DIR="${WORK_DIR}/JPEGImages"
ANNO_DIR="${WORK_DIR}/Annotations"
LOAD_DIR="${WORK_DIR}/load_data_dir"


echo "------------------------"
echo "Load data"
echo "source directory: ${LOAD_DIR}"
echo "destination directory: ${IMG_DIR} and ${ANNO_DIR}"

if ${PRE_CLR}; then
	echo "before loading, clear destination directory: ${IMG_DIR} and ${ANNO_DIR}"
	rm -rf ${IMG_DIR}/*
	rm -rf ${ANNO_DIR}/*
fi

python move_files.py \
	--src_dir=${LOAD_DIR} \
	--dest_dir_img=${IMG_DIR} \
	--dest_dir_anno=${ANNO_DIR}

echo "clear files in source directory: ${LOAD_DIR}"
rm -rf ${LOAD_DIR}/*
			
