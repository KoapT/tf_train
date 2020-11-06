# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exports trained model to TensorFlow frozen graph."""

import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from yolo.src.utils import param_file_access
from yolo.src import models_rigister

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/model_544_eval.ckpt',
                    'Checkpoint path')

flags.DEFINE_string('export_path', '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/model_544.pb',
                    'Path to output Tensorflow frozen graph.')

flags.DEFINE_string('model_info_path', '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/config.json',
                    'Model information file path.')

flags.DEFINE_string('sample_info_path', '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/LabelMap/label_map.json',
                    'Sample information file path.')


def main(unused_argv):
    if os.path.isfile(FLAGS.checkpoint_path + '.data-00000-of-00001'):
        ckpt_file_1 = FLAGS.checkpoint_path + '.index'
        ckpt_file_2 = FLAGS.checkpoint_path + '.meta'
        ckpt_file_3 = FLAGS.checkpoint_path + '.data-00000-of-00001'
        if (os.path.exists(ckpt_file_1) and
                os.path.exists(ckpt_file_2) and
                os.path.exists(ckpt_file_3)):
            checkpoint_path=FLAGS.checkpoint_path
        else:
            print('No ckpt file %s found!'%FLAGS.checkpoint_path)
            raise NameError
    elif os.path.isdir(FLAGS.checkpoint_path):
        try:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            checkpoint_path = os.path.join(FLAGS.checkpoint_path, ckpt_name)
        except:
            print('No ckpt file found in %s!' % FLAGS.checkpoint_path)
            raise NameError
    else:
        raise NameError
    print('loading model %s...\n'%checkpoint_path)

    model_info_path = FLAGS.model_info_path
    sample_info_path = FLAGS.sample_info_path

    config_path = model_info_path
    config_dict = param_file_access.get_json_params(config_path)
    model_name = config_dict['model_name']
    export_as_train_mode = False
    if 'export_config' in config_dict.keys():
        export_config = config_dict['export_config']
        if 'export_as_train_mode' in export_config.keys():
            export_as_train_mode = export_config['export_as_train_mode']

    model = models_rigister.get_model(model_name)
    model.initialize(
        model_info_path=model_info_path,
        sample_info_path=sample_info_path,
        is_training=True if export_as_train_mode else False)
    with tf.Graph().as_default() as graph:
        model.create_for_inference()
        output_names = ''
        split_sign = ','
        for name in model.get_output_names():
            output_names = output_names + name + split_sign
        output_names = output_names.rstrip(split_sign)
        # vars_list = tf.global_variables()
        # for var in vars_list:
        #     print(var)

        saver = tf.train.Saver(tf.model_variables())
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.export_path))
        freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=graph.as_graph_def(add_shapes=True),
            input_saver_def=saver.as_saver_def(),
            input_checkpoint=checkpoint_path,
            output_node_names=output_names,  # a string splited with ','
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=FLAGS.export_path,
            clear_devices=True,
            initializer_nodes=None)


if __name__ == '__main__':
    tf.app.run()
