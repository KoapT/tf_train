#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:55:45 2019

@author: rick
"""

import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('pipeline_config_path', '', 
                           'Pipeline.config file path')
tf.app.flags.DEFINE_integer('num_steps', 1, 
                            'Number of iteration steps')
tf.app.flags.DEFINE_string('fine_tune_checkpoint', '', 
                           'Fine tune model check point file path')
tf.app.flags.DEFINE_string('train_label_map_path', '', 
                           'Label map file path for training')
tf.app.flags.DEFINE_string('train_tf_record_path', '', 
                           'Tf .record file path for training')
tf.app.flags.DEFINE_string('eval_label_map_path', '', 
                           'Label map file path for evaluation')
tf.app.flags.DEFINE_string('eval_tf_record_path', '', 
                           'Tf .record file path for evaluation')
##for test=========
#FLAGS.num_steps = 1
#FLAGS.pipeline_config_path = '/home/rick/ImageRecognition/tensorFlowTrain/pretrain_models/ModelExampleFuncName_NetName/pipeline.config'
#FLAGS.fine_tune_checkpoint = '/home/rick/ImageRecognition/tensorFlowTrain/pretrain_models/ssdlite_mobilenetv2_infrared/model.ckpt'
#FLAGS.train_label_map_path = '/home/rick/ImageRecognition/tensorFlowTrain/samples_to_train/SampleExample/LabelMap/label_map.pbtxt'
#FLAGS.train_tf_record_path = '/home/rick/ImageRecognition/tensorFlowTrain/samples_to_train/Insulator/TFRecord/train.record'
#FLAGS.eval_label_map_path = '/home/rick/ImageRecognition/tensorFlowTrain/samples_to_train/Insulator/LabelMap/label_map.pbtxt'
#FLAGS.eval_tf_record_path = '/home/rick/ImageRecognition/tensorFlowTrain/samples_to_train/Insulator/TFRecord/eval.record'
##=============


print('modify: {}'.format(FLAGS.pipeline_config_path))
configs = config_util.get_configs_from_pipeline_file(FLAGS.pipeline_config_path)
pipeline_config = config_util.create_pipeline_proto_from_configs(configs)

#modify
num_classes = label_map_util.get_max_label_map_index(
        label_map_util.load_labelmap(FLAGS.train_label_map_path))
model_config = pipeline_config.model
getattr(model_config, model_config.WhichOneof('model')).num_classes = num_classes

train_config = pipeline_config.train_config
train_config.num_steps = FLAGS.num_steps
train_config.fine_tune_checkpoint = FLAGS.fine_tune_checkpoint

input_reader = pipeline_config.train_input_reader
input_reader.label_map_path = FLAGS.train_label_map_path
#input_path is a repeated
input_path = input_reader.tf_record_input_reader.input_path
for i in range(len(input_path)):
    input_path.remove(input_path[i])
input_path.append(FLAGS.train_tf_record_path)

#eval_input_reader is a repeated
if len(pipeline_config.eval_input_reader) != 0:
    input_reader = pipeline_config.eval_input_reader[0]
    input_reader.label_map_path = FLAGS.eval_label_map_path
    input_path = input_reader.tf_record_input_reader.input_path
    for i in range(len(input_path)):
        input_path.remove(input_path[i])
    input_path.append(FLAGS.eval_tf_record_path)
    input_reader = pipeline_config.eval_input_reader
    if len(input_reader) > 1:
        for i in range(1, len(input_reader)):
            input_reader.remove(input_reader[i])

#over write
directory = os.path.dirname(FLAGS.pipeline_config_path)
config_util.save_pipeline_config(pipeline_config, directory)