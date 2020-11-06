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

"""Provides data from semantic segmentation datasets.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow.

"""
import tensorflow as tf
from tensorflow.contrib import slim

dataset = slim.dataset
tfexample_decoder = slim.tfexample_decoder

KEY_NAME = 'name'
KEY_IMAGE = 'image'
KEY_IMAGE_FORMAT = 'image_format'
KEY_HEIGHT = 'height'
KEY_WIDTH = 'width'
KEY_MASK = 'mask'
KEY_MASK_FORMAT = 'mask_format'
KEY_DEPTH = 'depth'
KEY_DEPTH_FORMAT = 'depth_format'
KEY_LABEL = 'label'
KEY_LABEL_NAME = 'label_name'
KEY_OBJ_BOX = 'object_box'
KEY_OBJ_LABEL = 'object_label'
KEY_OBJ_LABEL_NAME = 'object_label_name'
KEY_OBJ_AREA = 'object_area'
KEY_OBJ_CROWD = 'object_is_crowd'
KEY_OBJ_DIFFICULT = 'object_difficult'
KEY_OBJ_GROUP = 'object_group_of'
KEY_OBJ_WEIGHT = 'object_weight'
KEY_OBJ_MASK = 'object_mask'



_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'mask': ('A semantic segmentation mask whose size matches image.'
             'Its value per pixel range from 0 (background) to num_classes.'),
    'depth': ('A depth image whose size matches image.'
              'Its value per pixel is depth range from 0~255.'),
}




def get_dataset(tfrecord_path, num_samples, 
                include_classify=False, 
                include_detect=False, 
                include_segment=False,
                include_depth=False):
  """Gets an instance of slim Dataset.

  Args:
    tfrecord_path: Tfrecord file path corresponding to list.

  Returns:
    An instance of slim Dataset.

  """
  # Specify how the TF-Examples are decoded.
  keys_to_features = {
      # Base content.
      'image/encoded': 
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/filename': 
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': 
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/height': 
          tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/width': 
          tf.FixedLenFeature((), tf.int64, default_value=0),
      # Image-level labels.
      'image/class/label':
          tf.VarLenFeature(tf.int64),
      'image/class/text':
          tf.VarLenFeature(tf.string),
      # Object boxes and classes.
      'image/object/bbox/xmin':
          tf.VarLenFeature(tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(tf.int64),
      'image/object/class/text':
          tf.VarLenFeature(tf.string),
      'image/object/area':
          tf.VarLenFeature(tf.float32),
      'image/object/is_crowd':
          tf.VarLenFeature(tf.int64),
      'image/object/difficult':
          tf.VarLenFeature(tf.int64),
      'image/object/group_of':
          tf.VarLenFeature(tf.int64),
      'image/object/weight':
          tf.VarLenFeature(tf.float32),
      'image/object/mask':
          tf.VarLenFeature(tf.string),
      # Segmentation mask.
      'image/segmentation/class/encoded': 
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/segmentation/class/format': 
          tf.FixedLenFeature((), tf.string, default_value='png'),
      # Depth.
      'image/depth/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/depth/format': 
          tf.FixedLenFeature((), tf.string, default_value='png'),
  }
      
  items_to_handlers = {
      KEY_NAME: tfexample_decoder.Tensor('image/filename'),
      KEY_HEIGHT: tfexample_decoder.Tensor('image/height'),
      KEY_WIDTH: tfexample_decoder.Tensor('image/width'),
      KEY_IMAGE: tfexample_decoder.Image(
              image_key='image/encoded',
              format_key='image/format',
              channels=3),
      KEY_IMAGE_FORMAT: tfexample_decoder.Tensor('image/format')
  }
  
  if include_classify:
      items_to_handlers[KEY_LABEL] = tfexample_decoder.Tensor(
              'image/class/label')
      items_to_handlers[KEY_LABEL_NAME] = tfexample_decoder.Tensor(
              'image/class/text', default_value='')
      
  if include_detect:
      items_to_handlers[KEY_OBJ_BOX] = tfexample_decoder.BoundingBox(
              keys=['ymin', 'xmin', 'ymax', 'xmax'], 
              prefix='image/object/bbox/')
      items_to_handlers[KEY_OBJ_LABEL] = tfexample_decoder.Tensor(
              'image/object/class/label')
      items_to_handlers[KEY_OBJ_LABEL_NAME] = tfexample_decoder.Tensor(
              'image/object/class/text', default_value='')

  if include_segment:
      items_to_handlers[KEY_MASK] = tfexample_decoder.Image(
              image_key='image/segmentation/class/encoded',
              format_key='image/segmentation/class/format',
              channels=1)
      items_to_handlers[KEY_MASK_FORMAT] = tfexample_decoder.Tensor(
              'image/segmentation/class/format')

  if include_depth:
      items_to_handlers[KEY_DEPTH] = tfexample_decoder.Image(
              image_key='image/depth/encoded',
              format_key='image/depth/format',
              channels=1)
      items_to_handlers[KEY_DEPTH_FORMAT] = tfexample_decoder.Tensor(
              'image/depth/format')

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=tfrecord_path,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
