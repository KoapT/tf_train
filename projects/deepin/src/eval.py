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
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""
import os
import math
import six
import tensorflow as tf

from deepin.src import models_rigister
from deepin.src.utils import param_file_access

slim = tf.contrib.slim
prefetch_queue = slim.prefetch_queue


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('is_show_log', True, 
                     'If show logs.')

# Settings for log directories.
flags.DEFINE_string('logdir', None, 
                    'Where to write the event logs.')
flags.DEFINE_string('checkpoint_dir', None, 
                    'Directory of model checkpoints.')

# Settings for evaluating the model.
flags.DEFINE_integer('eval_interval_secs', 60,
                     'How often (in seconds) to check if have a new checkpoint,'
                     'if have then run a evaluation.')
flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations(number of loops).'
                     'Will loop indefinitely upon nonpositive values.')

flags.DEFINE_string('model_info_path', None,
                    'Model information file path.')

# Sample dataset settings.
flags.DEFINE_string('sample_info_path', None,
                    'Sample information file path.')
flags.DEFINE_string('sample_list_path', 'Set/eval.txt',
                    'Sample list file path.')
flags.DEFINE_string('sample_path', 'TFRecord/eval.tfrecord',
                    'Sample path.')

flags.DEFINE_boolean('use_cpu', False,
                     'If use cpu.')
flags.DEFINE_boolean('allow_gpu_memory_dynamic_allocation', True,
                     'Allow gpu memory dynamic allocation')
flags.DEFINE_float('max_gpu_memory_usage', 0.9,
                   'Max usage of GPU memory.')





def main(unused_argv):
    if FLAGS.use_cpu:
        #run on cpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if FLAGS.is_show_log:
        tf.logging.set_verbosity(tf.logging.INFO)  #show log

    log_dir = FLAGS.logdir
    checkpoint_dir = FLAGS.checkpoint_dir
    model_info_path = FLAGS.model_info_path
#    model_config = param_file_access.get_json_params(model_info_path)
    sample_info_path = FLAGS.sample_info_path
    sample_list_path = FLAGS.sample_list_path
    sample_path = FLAGS.sample_path
    eval_interval_secs = FLAGS.eval_interval_secs
    max_number_of_evaluations = FLAGS.max_number_of_evaluations
    gpu_allow_growth = FLAGS.allow_gpu_memory_dynamic_allocation
    gpu_fraction = FLAGS.max_gpu_memory_usage
#    if 'eval_config' in model_config.keys():
#        eval_config = model_config['eval_config']
#        if 'eval_interval_secs' in eval_config.keys():
#            eval_interval_secs = eval_config['eval_interval_secs']
    
    num_samples = len(param_file_access.get_txt_params(sample_list_path))
    
    tf.gfile.MakeDirs(log_dir)
    
    # Create model.
    model_name = param_file_access.get_json_params(model_info_path)['model_name']
    model = models_rigister.get_model(model_name)
    model.initialize(
            model_info_path=model_info_path, 
            sample_info_path=sample_info_path, 
            is_training=False)
    batch_size = model.get_batch_size()
    
    with tf.Graph().as_default():
        #Build input
        input_queue = model.build_inputs(
                sample_path, num_samples, batch_size, num_clones=1)
        #Build model
        outputs = model.build_model(input_queue)
        train_item_list = outputs.train_item_list
        metric_dict = outputs.metric_dict
        
        if metric_dict:
            metrics_to_values, metrics_to_updates = (
                    tf.contrib.metrics.aggregate_metric_map(metric_dict))
            for metric_name, metric_value in six.iteritems(metrics_to_values):
                slim.summaries.add_scalar_summary(
                        metric_value, 'eval/%s'%metric_name, print_summary=True)
        
        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES)) 
        #add summaries for losses.
        for train_item in train_item_list:
            loss = train_item.loss
            summary_name = train_item.summary_name
            summaries.add(tf.summary.scalar('eval/%s_total_loss'%summary_name, loss))

        num_batches = int(math.ceil(num_samples / float(batch_size)))
        num_eval_iters = None
        if max_number_of_evaluations > 0:
            num_eval_iters = max_number_of_evaluations
        
        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(log_device_placement=False)
        session_config.gpu_options.allow_growth = gpu_allow_growth
        session_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        
        print('begin evaluation===========')
        slim.evaluation.evaluation_loop(
                master='',
                checkpoint_dir=checkpoint_dir,
                logdir=log_dir,
                num_evals=num_batches,
                eval_op=list(metrics_to_updates.values()) if metric_dict else None,
                max_number_of_evaluations=num_eval_iters,
                eval_interval_secs=eval_interval_secs,
                session_config=session_config)



if __name__ == '__main__':
    tf.app.run()
