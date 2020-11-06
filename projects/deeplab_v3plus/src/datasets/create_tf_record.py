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

"""Converts standard data to TFRecord file format with Example protos.

standard dataset is expected to have the following directory structure:

  + sample name
    + Annotations
    + JPEGImages
    + LabelMap
    + Set
    + TFRecord

Image folder:
  JPEGImages

Semantic segmentation annotations:
  Annotations

list folder:
  Set

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import tensorflow as tf
import build_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
        'image_folder',
        'JPEGImages',
        'Folder containing images.')

tf.app.flags.DEFINE_string(
        'label_folder',
        'Annotations',
        'Folder containing semantic segmentation labels.')

tf.app.flags.DEFINE_string(
        'list_path',
        'Set/eval.txt',
        'Path of list for training or evaluation.')

tf.app.flags.DEFINE_string(
        'output_dir',
        'TFRecord',
        'Path to save converted TFRecord file of TensorFlow examples.')


_NUM_SHARDS = 1



def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format.
    
    Args:
        dataset_split: The dataset split (e.g., train, eval).
    
    Raises:
        RuntimeError: If loaded image and label have different shape.
    """
    dataset = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset)
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
#        output_filename = os.path.join(FLAGS.output_dir,
#            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
        output_filename = os.path.join(FLAGS.output_dir, '%s.record' % (dataset))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % 
                                 (i + 1, len(filenames), shard_id))
                sys.stdout.flush()
                # Read the image.
                image_filename = os.path.join(
                    FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_filename = os.path.join(
                    FLAGS.label_folder, filenames[i] + '.' + FLAGS.label_format)
                seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                  raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, filenames[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
    _convert_dataset(FLAGS.list_path)


if __name__ == '__main__':
    tf.app.run()
