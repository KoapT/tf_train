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
"""Wrapper for providing semantic segmentation data."""
import tensorflow as tf
from deeplab import common



    

#def _load_image_into_tensors(image_path, crop_size):
#    name = os.path.splitext(os.path.basename(image_path))[0]
#    image = Image.open(image_path)
#    newHeight = crop_size[0]
#    newWidth = crop_size[1]
#    image = image.resize((newWidth, newHeight), Image.ANTIALIAS)
#    (width, height) = image.size
#    image_np = np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)
#    tensor_image = tf.convert_to_tensor(image_np)
#    tensor_height = tf.convert_to_tensor(height, tf.int64)
#    tensor_width = tf.convert_to_tensor(width, tf.int64)
#    tensor_name = tf.convert_to_tensor(name, tf.string)
#    return (tensor_image, tensor_height, tensor_width, tensor_name)

def _build_feed_tensors():
    channels = 3
    inputs_image = tf.placeholder(
            dtype=tf.uint8, 
            shape=[None, None, channels])
    inputs_height = tf.placeholder(
            dtype=tf.int32,
            shape=[1])
    inputs_width = tf.placeholder(
            dtype=tf.int32,
            shape=[1])
    inputs_name = tf.placeholder(
            dtype=tf.string,
            shape=[1])
    return (inputs_image, inputs_height, inputs_width, inputs_name)
    

def get():
    (inputs_image, inputs_height, 
     inputs_width, inputs_name) = _build_feed_tensors()
    
    inputs = {
            common.IMAGE: inputs_image,
            common.IMAGE_NAME: inputs_name,
            common.HEIGHT: inputs_height,
            common.WIDTH: inputs_width}

    return inputs
