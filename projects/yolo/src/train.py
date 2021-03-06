#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:13:33 2019

@author: rick
"""
import os
import time
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io
from tensorflow.contrib import slim

from yolo.src import models_rigister
from yolo.src.deployment import model_deploy
from yolo.src.utils import param_file_access, model_utils

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_clones', 2,
                     'Number of clones to deploy.')

flags.DEFINE_string('model_info_path', '',
                    'Model information file path.')

# Sample dataset settings.
flags.DEFINE_string('sample_info_path', '',
                    'Sample information file path.')  # label_map path
flags.DEFINE_string('sample_list_path', '',
                    'Sample list file path.')
flags.DEFINE_string('sample_path', '',
                    'Sample path.')

# Settings for logging.
flags.DEFINE_string('logdir', '',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.'
                     'Contain globel step and loss value.')
flags.DEFINE_integer('save_interval_secs', 600,
                     'How often, in seconds, we save the model checkpoint to disk.')
flags.DEFINE_integer('save_summaries_secs', 60,
                     'How often, in seconds, we compute the summaries.')

# Settings for fine-tuning the network.
flags.DEFINE_string('tf_initial_checkpoint_path', '',
                    'The path of initial checkpoint in tensorflow format.'
                    'If is None, it will create a new initial model checkpoint in logdir')
flags.DEFINE_boolean('initialize_last_layer_from_checkpoint', True,
                     'Do initialize last layer from check point')

flags.DEFINE_integer('training_number_of_steps', 500000,
                     'The number of steps used for training')

flags.DEFINE_integer('rest_interval_secs', 700,
                     'How often, in seconds, train progress take a rest.'
                     'If the value > 0, than train with rest.'
                     'It will be valid when newer checkpoint has been saved to disk,'
                     'so it must > checkpoint saving period.')
flags.DEFINE_integer('once_rest_secs', 60,
                     'How long for rest once in seconds.')

SCOPE = 'train'
NAME_IMAGE = 'image'
NAME_HEIGHT = 'height'
NAME_WIDTH = 'width'
NAME_LABEL = 'label'
NAME_PREDICT = 'predict'

model = None
begin_to_train_time = None


# can be debug with this function, it will be callback on every step
def train_step(sess, train_op, global_step, train_step_kwargs):
    """Function that takes a gradient step and specifies whether to stop.

    Args:
      sess: The current session.
      train_op: An `Operation` that evaluates the gradients and returns the
        total loss.
      global_step: A `Tensor` representing the global training step.
      train_step_kwargs: A dictionary of keyword arguments.

    Returns:
      The total loss and a boolean indicating whether or not to stop training.

    Raises:
      ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
    """
    start_time = time.time()

    trace_run_options = None
    run_metadata = None
    if 'should_trace' in train_step_kwargs:
        if 'logdir' not in train_step_kwargs:
            raise ValueError('logdir must be present in train_step_kwargs when should_trace is present')
        if sess.run(train_step_kwargs['should_trace']):
            trace_run_options = config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()

    # debug by user==============================
    global model
    if model is not None:
        model.every_before_train_step_callback_fn(sess)
        if isinstance(train_op, list):
            np_global_step = sess.run(global_step)  # previously get step before this iteration
            train_op = model.schedule_per_train_step(train_op, np_global_step)

    if isinstance(train_op, list):
        train_op.append(global_step)
        total_loss = sess.run(train_op,
                              options=trace_run_options,
                              run_metadata=run_metadata)
        np_global_step = total_loss.pop()
        train_op.pop()
    else:
        total_loss, np_global_step = sess.run([train_op, global_step],
                                              options=trace_run_options,
                                              run_metadata=run_metadata)

    if model is not None:
        model.every_after_train_step_callback_fn(sess)
    # ===========================================

    time_elapsed = time.time() - start_time

    if run_metadata is not None:
        print('run_metadata is not None')
        tl = timeline.Timeline(run_metadata.step_stats)
        trace = tl.generate_chrome_trace_format()
        trace_filename = os.path.join(train_step_kwargs['logdir'],
                                      'tf_trace-%d.json' % np_global_step)
        logging.info('Writing trace to %s', trace_filename)
        file_io.write_string_to_file(trace_filename, trace)
        if 'summary_writer' in train_step_kwargs:
            print('summary_writer is in train_step_kwargs')
            train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                                 'run_metadata-%d' %
                                                                 np_global_step)

    if 'should_log' in train_step_kwargs:
        if sess.run(train_step_kwargs['should_log']):
            if isinstance(total_loss, list):
                for total_loss_t in total_loss:
                    logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                                 np_global_step, total_loss_t, time_elapsed)
            else:
                logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                             np_global_step, total_loss, time_elapsed)

    # TODO(nsilberman): figure out why we can't put this into sess.run. The
    # issue right now is that the stop check depends on the global step. The
    # increment of global step often happens via the train op, which used
    # created using optimizer.apply_gradients.
    #
    # Since running `train_op` causes the global step to be incremented, one
    # would expected that using a control dependency would allow the
    # should_stop check to be run in the same session.run call:
    #
    #   with ops.control_dependencies([train_op]):
    #     should_stop_op = ...
    #
    # However, this actually seems not to work on certain platforms.
    if 'should_stop' in train_step_kwargs:
        should_stop = sess.run(train_step_kwargs['should_stop'])
    else:
        should_stop = False

    global begin_to_train_time
    if (begin_to_train_time is not None and FLAGS.rest_interval_secs > 0):
        train_duration = time.time() - begin_to_train_time
        rest_interval_secs = (FLAGS.rest_interval_secs
                              if FLAGS.rest_interval_secs > FLAGS.save_interval_secs else
                              FLAGS.save_interval_secs + 30)
        if train_duration > rest_interval_secs:
            logging.info('train take a rest')
            should_stop = True
    return total_loss, should_stop


def main(unused_argv):
    global model
    tf.logging.set_verbosity(tf.logging.INFO)  # show log

    initial_checkpoint_path = FLAGS.tf_initial_checkpoint_path
    ckpt_file_1 = os.path.join(
        os.path.dirname(initial_checkpoint_path), 'checkpoint')
    ckpt_file_2 = initial_checkpoint_path + '.index'
    ckpt_file_3 = initial_checkpoint_path + '.meta'
    ckpt_file_4 = initial_checkpoint_path + '.data-00000-of-00001'
    if not (os.path.exists(ckpt_file_1) and
            os.path.exists(ckpt_file_2) and
            os.path.exists(ckpt_file_3) and
            os.path.exists(ckpt_file_4)):
        initial_checkpoint_path = None
    print('initial_checkpoint_path:', initial_checkpoint_path)
    model_info_path = FLAGS.model_info_path
    model_config = param_file_access.get_json_params(model_info_path)
    sample_info_path = FLAGS.sample_info_path
    sample_list_path = FLAGS.sample_list_path
    sample_path = FLAGS.sample_path
    train_log_dir = FLAGS.logdir
    num_clones = FLAGS.num_clones
    clone_on_cpu = False
    gpu_allow_growth = True
    gpu_fraction = .92
    train_number_of_steps = FLAGS.training_number_of_steps
    log_every_n_steps = FLAGS.log_steps
    save_summaries_every_n_secs = FLAGS.save_summaries_secs
    save_ckpt_every_n_secs = FLAGS.save_interval_secs
    initialize_last_layer_from_checkpoint = FLAGS.initialize_last_layer_from_checkpoint

    print('num_clones:{}'.format(num_clones))
    print('initialize_last_layer_from_checkpoint:{}'.format(initialize_last_layer_from_checkpoint))

    num_samples = len(param_file_access.get_txt_params(sample_list_path))
    tf.gfile.MakeDirs(train_log_dir)

    # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
    config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu)

    # Create model.
    model_name = model_config['model_name']
    model = models_rigister.get_model(model_name)
    model.initialize(
        model_info_path=model_info_path,
        sample_info_path=sample_info_path,  # label_map path
        is_training=True)
    batch_size = model.get_batch_size()

    # Split the batch across GPUs.
    assert batch_size % config.num_clones == 0, (
        'Training batch size not divisble by number of clones (GPUs).')
    clone_batch_size = batch_size // config.num_clones
    with tf.Graph().as_default():
        # Define the inputs.
        with tf.device(config.inputs_device()):
            input_queue = model.build_inputs(
                sample_path, num_samples, clone_batch_size, config.num_clones)

        # Define the model and create clones.
        with tf.device(config.variables_device()):
            model_fn = model.build_model
            model_args = [input_queue]
            clones = model_deploy.create_clones(config, model_fn, args=model_args)

        # Define the optimizer based on the device specification.
        with tf.device(config.optimizer_device()):
            outputs = model.build_optimizer(train_number_of_steps)
            optimizer_item_list = outputs.optimizer_item_list

        # Gather summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # add summaries for learning rates.
        for optimizer_item in optimizer_item_list:
            summaries.add(tf.summary.scalar(
                ('train/{}_learn_rate'.format(optimizer_item.summary_name)
                 if optimizer_item.summary_name else
                 'train/learn_rate'),
                optimizer_item.learn_rate))
        # add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # last layers
        detection_layers = model.get_extra_layer_scopes()
        model_last_layers = []
        for var in tf.model_variables():
            for detection_layer in detection_layers:
                if detection_layer in var.name:
                    model_last_layers.append(var.name)
        # print('model_last_layers:', model_last_layers)
        # print('yolo vars:', tf.trainable_variables(scope='yolo'))
        # print('neck vars:', tf.trainable_variables(scope='neck'))

        with tf.device(config.variables_device()):
            # Create the global step on the device storing the variables.
            global_step = tf.train.get_or_create_global_step()

            train_tensor_list = []
            num_optimizers = len(optimizer_item_list)
            for i in range(num_optimizers):
                optimizer = optimizer_item_list[i].optimizer
                summary_name = optimizer_item_list[i].summary_name
                clone_scope_list = []
                clone_device_list = []
                clone_loss_list = []
                clone_varlist_list = []
                for clone in clones:
                    clone_scope_list.append(clone.scope)
                    clone_device_list.append(clone.device)
                    clone_loss_list.append(clone.outputs.train_item_list[i].loss)
                    # clone_varlist_list.append(clone.outputs.train_item_list[i].var_list)
                    # 固定backbone
                    var_list_to_upgrade = tf.trainable_variables(scope='yolo')
                    clone_varlist_list.append(var_list_to_upgrade)

                total_loss, total_grads_and_vars = model_deploy.optimize_clones_custom(
                    optimizer=optimizer,
                    clone_scope_list=clone_scope_list,
                    clone_device_list=clone_device_list,
                    clone_loss_list=clone_loss_list,
                    clone_varlist_list=clone_varlist_list)
                # total_grads_and_vars_clip = [(tf.clip_by_value(grad, -1., 1.),var) for grad,var in total_grads_and_vars]
                total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
                # Create gradient update op.
                grad_updates = optimizer.apply_gradients(
                    total_grads_and_vars, global_step=global_step)
                summaries.add(tf.summary.scalar(
                    ('train/{}_total_loss'.format(summary_name)
                     if summary_name else
                     'train/total_loss'),
                    total_loss))
                first_clone_scope = clones[0].scope
                # Gather update_ops from the first clone. These contain, for example,
                # the updates for the batch_norm variables created by model_fn.
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

                # Create gradient update op.
                update_ops.append(grad_updates)
                update_op = tf.group(*update_ops)
                with tf.control_dependencies([update_op]):
                    train_tensor = tf.identity(total_loss, name='train_op')
                    train_tensor_list.append(train_tensor)
        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones().

        summaries |= set(
            tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = gpu_allow_growth
        session_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

        print('begin train===========')


        global begin_to_train_time
        begin_to_train_time = time.time()

        # Start the training.
        slim.learning.train(
            train_op=train_tensor_list,
            train_step_fn=train_step,
            logdir=train_log_dir,
            log_every_n_steps=log_every_n_steps,
            number_of_steps=train_number_of_steps,
            session_config=session_config,
            init_fn=model_utils.get_model_init_fn(
                train_log_dir,
                initial_checkpoint_path,
                initialize_last_layer=initialize_last_layer_from_checkpoint,
                last_layers=model_last_layers,
                ignore_missing_vars=True),
            summary_op=summary_op,
            save_summaries_secs=save_summaries_every_n_secs,
            save_interval_secs=save_ckpt_every_n_secs)


if __name__ == '__main__':
    tf.app.run()
