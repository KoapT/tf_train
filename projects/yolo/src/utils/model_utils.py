#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:48:17 2019

@author: rick
"""
import os
import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim




def logit_to_multi_label(logits, threshold):
    '''
    Args: 
        logits, the tensor with shape [batch, num_classes],
            e.g. [[0.2,0.6,0.9],[0.3,0.1,0.8],...].
        threshold, if logits > threshold, than will be set, 
            else will be clear.
    Return:
        labels, the tensor with shape [batch, num_classes],
            [num_classes] is a label list, non-zero bit means that have a label. 
    '''
    label_list = tf.where(tf.greater(logits, threshold))
    logits_shape = tf.shape(logits)
    def _func_np(label_list_np, logits_shape_np):
        batch = logits_shape_np[0]
        max_num_classes = logits_shape_np[1]
        labels = np.zeros((batch, max_num_classes), np.int32)
        cnts = np.zeros((batch), np.int32)
        for i in range(np.shape(label_list_np)[0]):
            b = label_list_np[i, 0]
            l = label_list_np[i, 1]
            labels[b, cnts[b]] = l
            cnts[b] = cnts[b] + 1
        return labels
    labels = tf.py_func(_func_np, [label_list, logits_shape], tf.int32)
    return labels

def logit_to_label(logits):
    '''
    Args:
        logits is the tensor with shape
            [batch, shape(e.g. height*width or count), channels] or [batch, channels],
            channel indices is 0,1,2,3,4,5,...,num_classes-1
            respected to label value,
            in one point(e.g. x,y) channel[the label mask value(channel indices)]
            has the max value;
    Return:
        labels is the mask with value 0,1,2,3,4,5,...,num_classes-1
            and with shape [batch, shape(e.g. height*width or count)] or [batch];
    '''
    labels = tf.argmax(logits, axis=tf.size(tf.shape(logits))-1)
    labels = tf.cast(labels, tf.int32)
    return labels
    
def label_to_logit(labels, num_classes, on_value=1.0, off_value=0.0):
    '''
    Args:
        labels is the mask with value 0,1,2,3,4,5,...,num_classes-1
            and with shape [batch, shape(e.g. height*width or count)] or [batch];
        num_classes is the total number 
            with backgound class + original classes;
    Return:
        logits is the tensor with shape 
            [batch, shape(e.g. height*width or count), channels],
            channel indices is 0,1,2,3,4,5,...,num_classes-1
            respected to label value,
            in one point(e.g. x,y) channel[the label mask value(channel indices)]==1 else ==0;
    '''
    logits = slim.one_hot_encoding(
            tf.cast(labels, dtype=tf.int32), 
            num_classes, on_value=on_value, off_value=off_value)
    return logits

def add_softmax_cross_entropy_loss(logits,
                                   labels,
                                   class_id_list,
                                   scope=None):
    """Adds softmax cross entropy loss for logits of each scale.
    
    Args:
        logits: The logits have shape [batch, shape, num_classes].
        labels: Groundtruth labels with shape [batch, shape, 1] or [batch, shape].
        class_id_list: Class id list.
        scope: String, the scope for the loss.
    
    Raises:
        ValueError: Label or logits is None.
    """
    if logits is None:
        raise ValueError('No logit for softmax cross entropy loss.')
    if labels is None:
        raise ValueError('No label for softmax cross entropy loss.')
    if class_id_list is None:
        raise ValueError('No class id list for softmax cross entropy loss.')
    if len(class_id_list) == 0:
        raise ValueError('Class id list is empty for softmax cross entropy loss.')

    weights = tf.equal(labels, class_id_list[0])
    for i in range(1, len(class_id_list)):
        weights = weights | tf.equal(labels, class_id_list[i])
    weights = tf.to_float(weights)
    
    num_classes = len(class_id_list)
    label_logits = label_to_logit(labels, num_classes)
    label_logits = tf.reshape(label_logits, shape=[-1, num_classes])
    logits = tf.reshape(logits, shape=[-1, num_classes])
    weights = tf.reshape(weights, shape=[-1])
    loss = tf.losses.softmax_cross_entropy(
            label_logits, logits,
            weights=weights,
            scope=scope)
    return loss

def add_sigmoid_cross_entropy_loss(logits,
                                   labels,
                                   class_id_list,
                                   scope=None):
    """Adds sigmoid cross entropy loss for logits of each scale.
    
    Args:
        logits: The logits have shape [batch, shape, num_classes].
        labels: Groundtruth labels with shape [batch, shape, 1] or [batch, shape].
        class_id_list: Class id list.
        scope: String, the scope for the loss.
    
    Raises:
        ValueError: Label or logits is None.
    """
    if logits is None:
        raise ValueError('No logit for softmax cross entropy loss.')
    if labels is None:
        raise ValueError('No label for softmax cross entropy loss.')
    if class_id_list is None:
        raise ValueError('No class id list for softmax cross entropy loss.')
    if len(class_id_list) == 0:
        raise ValueError('Class id list is empty for softmax cross entropy loss.')

    weights = tf.equal(labels, class_id_list[0])
    for i in range(1, len(class_id_list)):
        weights = weights | tf.equal(labels, class_id_list[i])
    weights = tf.to_float(weights)
    
    num_classes = len(class_id_list)
    label_logits = label_to_logit(labels, num_classes)
    label_logits = tf.reshape(label_logits, shape=[-1, num_classes])
    logits = tf.reshape(logits, shape=[-1, num_classes])
    weights = tf.reshape(weights, shape=[-1])
    loss = tf.losses.sigmoid_cross_entropy(
            label_logits, logits,
            weights=1.0,
            scope=scope)
    return loss

def get_model_learning_rate(
    learning_policy, base_learning_rate, learning_rate_decay_step,
    learning_rate_decay_factor, training_number_of_steps, learning_power,
    slow_start_step, slow_start_learning_rate, end_learning_rate=0.0):
  """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy is not recognized.
  """
  global_step = tf.train.get_or_create_global_step()
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        learning_rate=base_learning_rate,
        global_step=global_step,
        decay_steps=learning_rate_decay_step,
        decay_rate=learning_rate_decay_factor,
        staircase=True,
        name='step_decay_learning_rate')
    learning_rate = tf.where(learning_rate < end_learning_rate, end_learning_rate,
                             learning_rate)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        learning_rate=base_learning_rate,
        global_step=global_step,
        decay_steps=training_number_of_steps,
        end_learning_rate=end_learning_rate,
        power=learning_power,
        cycle=False,
        name='polynomial_decay_learning_rate')
  elif learning_policy == 'fixed':
      learning_rate = tf.constant(base_learning_rate, name='fixed_learning_rate')
  else:
    raise ValueError('Unknown learning policy.')

  # Employ small learning rate at the first few steps for warm start.
  warm_up_learning_rate = tf.add(slow_start_learning_rate,
                              (base_learning_rate-slow_start_learning_rate)/slow_start_step*tf.cast(global_step,tf.float32),
                              name='warm_up_learning_rate')
  return tf.where(global_step < slow_start_step, warm_up_learning_rate,
                  learning_rate)

def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer=True,
                      last_layers=None,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  exclude_list = ['global_step']
  if (not initialize_last_layer) and (last_layers is not None):
    exclude_list.extend(last_layers)

  variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

  if variables_to_restore:
    return slim.assign_from_checkpoint_fn(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
  return None

def save_model(sess, save_dir, ckpt_name='model.ckpt', save_step=None):
    path = os.path.join(save_dir, ckpt_name)
    saver = tf.train.Saver()
    save_path = saver.save(sess, path, global_step=save_step)
    return save_path

def restore_model(sess, restore_dir, ckpt_name='model.ckpt'):
    path = os.path.join(restore_dir, ckpt_name+'.meta')
    saver = tf.train.import_meta_graph(path)
    saver.restore(sess, tf.train.latest_checkpoint(restore_dir))
    
def copytensor(tensor):
    return tf.add(tensor, tf.zeros_like(tensor))

def summariyFeaturemapByName(tenor_name, summary_name, max_outputs=None):
    '''Tensor shape must be [batch_size, height, width, channels],
       only summary first feature map in a batch.'''
    featuremap = tf.get_default_graph().get_tensor_by_name(tenor_name)
    channels = featuremap.get_shape().as_list()[3]
    features = []
    for i in range(channels):
        feature = tf.expand_dims(featuremap[0, :, :, i], axis=2)
        features.append(feature)
    featuremap_summary = tf.stack(features, axis=0)
    tf.summary.image(
            summary_name, featuremap_summary, 
            max_outputs=channels if max_outputs is None else max_outputs)

def get_label_weights(labels, class_id_list):
    '''Labels in class_id_list respect to weight 1.0, 
        else will respect to weight 0.0.
       return weights have the same shape of labels.
    '''
    num_classes = len(class_id_list)
    weights_mask = tf.equal(labels, class_id_list[0])
    for i in range(1, num_classes):
        weights_mask = weights_mask | tf.equal(labels, class_id_list[i])
    weights = tf.to_float(weights_mask)
    return weights

def filter_labels(labels, class_id_list):
    '''Labels in class_id_list will be reserved, else will be set to 0,
        with data type int32. So label 0(class id) should be background.
    '''
    labels = tf.cast(labels, dtype=tf.int32)
    mask = tf.equal(labels, class_id_list[0])
    for i in range(1, len(class_id_list)):
        mask = mask | tf.equal(labels, class_id_list[i])
    labels_filtered = tf.where(mask, labels, tf.zeros_like(labels, dtype=tf.int32))
    return labels_filtered

def build_tensor_map(key_list_tensor, value_list_tensor, default_value):
    return tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            keys=key_list_tensor,
            values=value_list_tensor),
        default_value=default_value)


def mapping_tensor(input_key_tensor, tensor_map):
    '''
    e.g. input_key_tensor=tensor([b'person', b'aeroplane', b'']),
            then output_value_tensor=tensor([15, 1, -1]).
    '''
    # tensor_map.init.run()
    output_value_tensor = tensor_map.lookup(input_key_tensor)
    return output_value_tensor