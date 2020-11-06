#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:34:48 2019

@author: rick
"""

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from yolo.src.core import inputs
from yolo.src.datasets.std import std_tfrecord_dataset
from yolo.src.utils import shape_utils
import tensorflow.contrib.slim as slim

dataset_data_provider = slim.dataset_data_provider
prefetch_queue = slim.prefetch_queue


NAME = 'name'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE = 'image'
IMAGE_FORMAT = 'image_format'
MASK = 'mask'
MASK_FORMAT = 'mask_format'
DEPTH = 'depth'
DEPTH_FORMAT = 'depth_format'
NUM_LABELS = 'number_labels'
LABEL = 'label'
LABEL_NAME = 'label_name'
NUM_OBJS = 'number_objects'
OBJ_BOX = 'object_box'
OBJ_LABEL = 'object_label'
OBJ_LABEL_NAME = 'object_label_name'


ERRO_EXCEED_MAX_LABELS = 'exceed_max_labels_erro'
ERRO_EXCEED_MAX_OBJS = 'exceed_max_objects_erro'


class StdTFRecordInputs(inputs.Inputs):
    def _build_data(self, data_provider, max_num_labels, max_num_objects):
        tf.logging.set_verbosity(tf.logging.INFO)
        (
         name, 
         height, 
         width, 
         image,
         image_format
         ) = data_provider.get(
                [
                 std_tfrecord_dataset.KEY_NAME,
                 std_tfrecord_dataset.KEY_HEIGHT,
                 std_tfrecord_dataset.KEY_WIDTH,
                 std_tfrecord_dataset.KEY_IMAGE,
                 std_tfrecord_dataset.KEY_IMAGE_FORMAT,
                ])
    
        if image.shape.ndims != 3:
            raise ValueError('Input image shape must be [height,width,3].')
        
        num_labels = None
        label = None
        label_name = None
        num_objs = None
        obj_label = None 
        obj_label_name = None 
        obj_box = None
        mask = None
        mask_format = None
        depth = None
        depth_format = None
        
        if std_tfrecord_dataset.KEY_LABEL in data_provider.list_items():
            label, = data_provider.get([std_tfrecord_dataset.KEY_LABEL])
            num_labels = tf.shape(label)[0]
            label_plus1 = tf.add(label, 1)
            label_t = shape_utils.pad_or_clip_nd(label_plus1, [max_num_labels])
            label_t = tf.subtract(label_t, 1)
            with tf.control_dependencies([tf.assert_less_equal(
                    num_labels, max_num_labels,
                    message=ERRO_EXCEED_MAX_LABELS)]):
                label = tf.identity(label_t)
        
        if std_tfrecord_dataset.KEY_LABEL_NAME in data_provider.list_items():
            label_name, = data_provider.get([std_tfrecord_dataset.KEY_LABEL_NAME])
            label_name = shape_utils.pad_or_clip_nd(label_name, [max_num_labels])
        
        if std_tfrecord_dataset.KEY_OBJ_LABEL in data_provider.list_items():
            obj_label, = data_provider.get([std_tfrecord_dataset.KEY_OBJ_LABEL])
            num_objs = tf.shape(obj_label)[0]
            obj_label_plus1 = tf.add(obj_label, 1)
            obj_label_t = shape_utils.pad_or_clip_nd(obj_label_plus1, [max_num_objects])
            obj_label_t = tf.subtract(obj_label_t, 1)
            with tf.control_dependencies([tf.assert_less_equal(
                    num_objs, max_num_objects,
                    message=ERRO_EXCEED_MAX_OBJS)]):
                obj_label = tf.identity(obj_label_t)
            
        
        if std_tfrecord_dataset.KEY_OBJ_LABEL_NAME in data_provider.list_items():
            obj_label_name, = data_provider.get([std_tfrecord_dataset.KEY_OBJ_LABEL_NAME])
            num_objs = tf.shape(obj_label_name)[0]
            obj_label_name = shape_utils.pad_or_clip_nd(obj_label_name, [max_num_objects])
        
        if std_tfrecord_dataset.KEY_OBJ_BOX in data_provider.list_items():
            obj_box, = data_provider.get([std_tfrecord_dataset.KEY_OBJ_BOX])
            obj_box = shape_utils.pad_or_clip_nd(obj_box, [max_num_objects, 4])
        
        if std_tfrecord_dataset.KEY_MASK in data_provider.list_items():
            mask, = data_provider.get([std_tfrecord_dataset.KEY_MASK])
            mask_format, = data_provider.get([std_tfrecord_dataset.KEY_MASK_FORMAT])
            if mask.shape.ndims == 2:
               mask = tf.expand_dims(mask, 2)
            elif mask.shape.ndims == 3 and mask.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input mask shape must be [height,width], or [height, width, 1].')
        
        if std_tfrecord_dataset.KEY_DEPTH in data_provider.list_items():
            depth, = data_provider.get([std_tfrecord_dataset.KEY_DEPTH])
            depth_format, = data_provider.get([std_tfrecord_dataset.KEY_DEPTH_FORMAT])
            if depth.shape.ndims == 2:
               depth = tf.expand_dims(depth, 2)
            elif depth.shape.ndims == 3 and depth.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input depth shape must be [height,width], or [height, width, 1].')
        
        return (name, height, width, image, image_format, 
                mask, mask_format, depth, depth_format,
                num_labels, label, label_name,
                num_objs, obj_box, obj_label, obj_label_name)
    
    def get(self,
            tfrecord_path,
            num_samples,
            batch_size,
            include_classify=False, 
            include_detect=False, 
            include_segment=False,
            include_depth=False,
            preprocess_fn=None,
            preprocess_args=None,
            preprocess_kwargs=None,
            num_clones=1,
            num_threads=1,
            max_num_batches=32,
            max_capacity_per_clone=128,
            allow_smaller_final_batch=False,
            dynamic_pad=True,
            is_shuffle=True,
            is_training=True,
            max_num_labels=20,
            max_num_objects=100):
        dataset = std_tfrecord_dataset.get_dataset(
                tfrecord_path, num_samples,
                include_classify=include_classify, 
                include_detect=include_detect, 
                include_segment=include_segment,
                include_depth=include_depth)
        data_provider = dataset_data_provider.DatasetDataProvider(
                dataset, 
                num_readers=1,
                num_epochs=None if is_training else 1,
                shuffle=is_shuffle)
        
        (name, height, width, image, image_format, 
         mask, mask_format, depth, depth_format,
         num_labels, label, label_name,
         num_objects, obj_box, obj_label, obj_label_name) = self._build_data(
                 data_provider, max_num_labels, max_num_objects)
        
        sample = {
                NAME: name,
                HEIGHT: height,
                WIDTH: width,
                IMAGE: image,
                IMAGE_FORMAT: image_format
            }
        
        if mask is not None:
           sample[MASK] = mask
        if mask_format is not None:
            sample[MASK_FORMAT] = mask_format
        if depth is not None:
            sample[DEPTH] = depth
        if depth_format is not None:
            sample[DEPTH_FORMAT] = depth_format
        if num_labels is not None:
            sample[NUM_LABELS] = num_labels
        if label is not None:
            sample[LABEL] = label
        if label_name is not None:
            sample[LABEL_NAME] = label_name
        if num_objects is not None:
            sample[NUM_OBJS] = num_objects
        if obj_box is not None:
            sample[OBJ_BOX] = obj_box
        if obj_label is not None:
            sample[OBJ_LABEL] = obj_label
        if obj_label_name is not None:
            sample[OBJ_LABEL_NAME] = obj_label_name
        
        preprocess_args = preprocess_args or []
        preprocess_kwargs = preprocess_kwargs or {}
        if preprocess_fn is not None:
            sample = preprocess_fn(sample, *preprocess_args, **preprocess_kwargs)
          
        samples = tf.train.batch(
                sample,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=max_num_batches*batch_size,
                allow_smaller_final_batch=allow_smaller_final_batch,  #if allow_smaller_final_batch==True, tensor batch size will be None, so cannot use prefetch_queue
                dynamic_pad=dynamic_pad)
        input_queue = prefetch_queue.prefetch_queue(
                samples, capacity=max_capacity_per_clone*num_clones)
#        samples = input_queue.dequeue()
        
        return input_queue

