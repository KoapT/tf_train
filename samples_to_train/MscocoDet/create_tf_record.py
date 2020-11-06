# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python create_tf_record.py (--set=train or eval, default is train)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import json

import PIL.Image
import tensorflow as tf

import dataset_util
import xml.etree.ElementTree as ET

flags = tf.app.flags
flags.DEFINE_boolean('ignore_difficult_instances', False,
                     'Whether to ignore difficult instances')

FLAGS = flags.FLAGS

CURRENT_DIR = os.path.abspath('.')
SET_TRAIN = 'train'
SET_EVAL = 'eval'

data_dir = CURRENT_DIR
images_dir = os.path.join(data_dir, 'JPEGImages')
annotations_dir = os.path.join(data_dir, 'Annotations')
label_map_path = os.path.join(data_dir, 'LabelMap', 'label_map.json')
set_train_path = os.path.join(data_dir, 'Set', SET_TRAIN + '.txt')
set_eval_path = os.path.join(data_dir, 'Set', SET_EVAL + '.txt')
output_train_path = os.path.join(data_dir, 'TFRecord', SET_TRAIN + '.record')
output_eval_path = os.path.join(data_dir, 'TFRecord', SET_EVAL + '.record')

def get_label_map_dict_from_json(label_map_path):
    label_map_dict = {}
    with open(label_map_path, 'r') as f:
        params = json.load(f)
    for param in params:
        label_map_dict[param['name']] = param['id']
    return label_map_dict


def dict_to_tf_example(anno_path, data,
                       image_directory,
                       label_map_dict,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      image_directory: Path to images
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    # img_path = os.path.join(image_directory, data['filename'])
    img_name = data['filename']
    if len(img_name.split('.')) == 1:
        img_path0 = data['path']
        ext_name = img_path0.split('.')[-1]
        img_name = img_name + '.' + ext_name
    img_path = os.path.join(image_directory, img_name)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG: {}'.format(img_name))
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            try:
                classes.append(label_map_dict[obj['name']])
            except KeyError as e:
                print('KeyError')
                print(anno_path)
                print(img_path)
                print(e)
                raise KeyError()
            # truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        # 'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example

def proccess(set_path, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    label_map_dict = get_label_map_dict_from_json(label_map_path)

    examples_path = set_path
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(annotations_dir, example + '.xml')
        with tf.gfile.GFile(path, 'r') as fid:
            xml = ET.parse(fid).getroot()
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example(path, data, images_dir, label_map_dict,
                                        FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main(_):
    print('start to create record according to txt of sample set...')
    print('data directory is: ' + data_dir)
    print('==========================')

    proccess(set_train_path, output_train_path)
    proccess(set_eval_path, output_eval_path)

    print('==========================')
    print('record generate done: ')
    print(output_train_path)
    print(output_eval_path)
    print('create record done!')


if __name__ == '__main__':
    tf.app.run()
