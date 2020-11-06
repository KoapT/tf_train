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
import sys
sys.path.append('..')
from utils import param_file_access
import tensorflow as tf

slim = tf.contrib.slim
dataset = slim.dataset
tfexample_decoder = slim.tfexample_decoder

flags = tf.app.flags
FLAGS = flags.FLAGS




_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}





def get_dataset(list_path, tfrecord_path, label_map_path, ignore_label=255):
  """Gets an instance of slim Dataset.

  Args:
    list_path: Path of sample list file.
    tfrecord_path: Tfrecord file path corresponding to list.
    label_map_path: Path of sample label map file.

  Returns:
    An instance of slim Dataset.

  """
  
  num_samples = len(param_file_access.get_txt_params(list_path))
  num_classes = len(param_file_access.get_json_params(label_map_path))


  # Specify how the TF-Examples are decoded.
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
  }
  items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      'labels_class': tfexample_decoder.Image(
          image_key='image/segmentation/class/encoded',
          format_key='image/segmentation/class/format',
          channels=1),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=tfrecord_path,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=num_classes,
      ignore_label=ignore_label,
      name='pascal_voc_seg',
      multi_label=True)
