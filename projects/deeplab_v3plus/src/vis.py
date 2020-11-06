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

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""
import math
import os.path
import time
import numpy as np
import tensorflow as tf
from deeplab import common
from deeplab import model
from datasets import segmentation_dataset
from inputs import input_generator
from deeplab.utils import save_annotation
import utils.param_file_access as param_reader
from PIL import Image

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('vis_crop_size', [513, 513],
                           'Crop size [height, width] for visualization.')

flags.DEFINE_integer('eval_interval_secs', 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.
flags.DEFINE_string('tfrecord_path', 'TFRecord/eval.tfrecord',
                    'Tfrecord file path.')

flags.DEFINE_string('label_map_path', 'LabelMap/label_map.json',
                    'Label map file path')

flags.DEFINE_string('list_path', 'Set/eval.txt',
                    'List file path.')

flags.DEFINE_string('config_path', 'pipeline.json',
                    'Configure file path.')

flags.DEFINE_string('vis_split', 'val',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_enum('colormap_type', 'pascal', ['pascal', 'cityscapes'],
                  'Visualization colormap type.')

flags.DEFINE_boolean('also_save_raw_predictions', True,
                     'Also save raw predictions.')

flags.DEFINE_integer('max_number_of_iterations', 0,
                     'Maximum number of visualization iterations. Will loop '
                     'indefinitely upon nonpositive values.')

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'



def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
  """Converts the predicted label for evaluation.

  There are cases where the training labels are not equal to the evaluation
  labels. This function is used to perform the conversion so that we could
  evaluate the results on the evaluation server.

  Args:
    prediction: Semantic segmentation prediction.
    train_id_to_eval_id: A list mapping from train id to evaluation id.

  Returns:
    Semantic segmentation prediction whose labels have been changed.
  """
  converted_prediction = prediction.copy()
  for train_id, eval_id in enumerate(train_id_to_eval_id):
    converted_prediction[prediction == train_id] = eval_id

  return converted_prediction


def _process_batch(sess, original_images, semantic_predictions, image_names,
                   image_heights, image_widths, image_id_offset, image_resize,
                   resize_image_feed, resize_size_feed,
                   save_dir, raw_save_dir, train_id_to_eval_id=None):
  """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    original_images: One batch of original images.
    semantic_predictions: One batch of semantic segmentation predictions.
    image_names: Image names.
    image_heights: Image heights.
    image_widths: Image widths.
    image_id_offset: Image id offset for indexing images.
    save_dir: The directory where the predictions will be saved.
    raw_save_dir: The directory where the raw predictions will be saved.
    train_id_to_eval_id: A list mapping from train id to eval id.
  """
  (original_images,
   semantic_predictions,
   image_names,
   image_heights,
   image_widths) = sess.run([original_images, semantic_predictions,
                             image_names, image_heights, image_widths])

  num_image = semantic_predictions.shape[0]
  for i in range(num_image):
    image_height = np.squeeze(image_heights[i])
    image_width = np.squeeze(image_widths[i])
    original_image = np.squeeze(original_images[i])
    semantic_prediction = np.squeeze(semantic_predictions[i])
#    crop_semantic_prediction = semantic_prediction[:image_height, :image_width]
    crop_semantic_prediction = sess.run(
            image_resize, 
            feed_dict={resize_image_feed:semantic_prediction[:,:,np.newaxis], 
                       resize_size_feed:[image_height, image_width]})
    crop_semantic_prediction = np.squeeze(crop_semantic_prediction)

    # Save image.
    save_annotation.save_annotation(
        original_image, save_dir, _IMAGE_FORMAT % (image_id_offset + i),
        add_colormap=False)

    # Save prediction.
    save_annotation.save_annotation(
        crop_semantic_prediction, save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i), add_colormap=True,
        colormap_type=FLAGS.colormap_type)

    if FLAGS.also_save_raw_predictions:
      image_filename = os.path.basename(image_names[i])

      if train_id_to_eval_id is not None:
        crop_semantic_prediction = _convert_train_id_to_eval_id(
            crop_semantic_prediction,
            train_id_to_eval_id)
      save_annotation.save_annotation(
          crop_semantic_prediction, raw_save_dir, image_filename,
          add_colormap=False)


def main(unused_argv):
  cfg = param_reader.get_json_params(FLAGS.config_path)
  FLAGS.model_variant = cfg['model_variant']
  FLAGS.vis_crop_size = [cfg['crop_size_height'], cfg['crop_size_width']]
  FLAGS.max_number_of_iterations = cfg['max_number_of_iterations']
  FLAGS.output_stride = cfg['output_stride']   
  if 'atrous_rates' in cfg.keys():
      FLAGS.atrous_rates = cfg['atrous_rates']
  if 'decoder_output_stride' in cfg.keys():
      FLAGS.decoder_output_stride = cfg['decoder_output_stride']
  
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.list_path, FLAGS.tfrecord_path, FLAGS.label_map_path)
  train_id_to_eval_id = None

  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)
  save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(save_dir)
  raw_save_dir = os.path.join(
      FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(raw_save_dir)

  tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

  g = tf.Graph()
  with g.as_default():
    resize_image_feed = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
    resize_size_feed = tf.placeholder(dtype=tf.int32, shape=[2])
    image_resize = tf.image.resize_images(
            resize_image_feed, resize_size_feed, 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    
    samples = input_generator.get(dataset,
                                  FLAGS.vis_crop_size,
                                  FLAGS.vis_batch_size,
                                  min_resize_value=FLAGS.min_resize_value,
                                  max_resize_value=FLAGS.max_resize_value,
                                  resize_factor=FLAGS.resize_factor,
                                  dataset_split=FLAGS.vis_split,
                                  is_training=False,
                                  model_variant=FLAGS.model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.vis_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]

    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      # Only support batch_size = 1, since we assume the dimensions of original
      # image after tf.squeeze is [height, width, 3].
      assert FLAGS.vis_batch_size == 1

      # Reverse the resizing and padding operations performed in preprocessing.
      # First, we slice the valid regions (i.e., remove padded region) and then
      # we reisze the predictions back.
      original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
      original_image_shape = tf.shape(original_image)
      predictions = tf.slice(
          predictions,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=g,
                             logdir=FLAGS.vis_logdir,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=saver)
    num_batches = int(math.ceil(
        dataset.num_samples / float(FLAGS.vis_batch_size)))
    last_checkpoint = None

    # Loop to visualize the results when new checkpoint is created.
    num_iters = 0
    while (FLAGS.max_number_of_iterations <= 0 or
           num_iters < FLAGS.max_number_of_iterations):
      num_iters += 1
      last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
          FLAGS.checkpoint_dir, last_checkpoint)
      start = time.time()
      tf.logging.info(
          'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      tf.logging.info('Visualizing with model %s', last_checkpoint)

      with sv.managed_session(FLAGS.master,
                              start_standard_services=False) as sess:
        sv.start_queue_runners(sess)
        sv.saver.restore(sess, last_checkpoint)

        image_id_offset = 0
        for batch in range(num_batches):
          tf.logging.info('Visualizing batch %d / %d', batch + 1, num_batches)
          _process_batch(sess=sess,
                         original_images=samples[common.ORIGINAL_IMAGE],
                         semantic_predictions=predictions,
                         image_names=samples[common.IMAGE_NAME],
                         image_heights=samples[common.HEIGHT],
                         image_widths=samples[common.WIDTH],
                         image_id_offset=image_id_offset,
                         image_resize=image_resize,
                         resize_image_feed=resize_image_feed,
                         resize_size_feed=resize_size_feed,
                         save_dir=save_dir,
                         raw_save_dir=raw_save_dir,
                         train_id_to_eval_id=train_id_to_eval_id)
          image_id_offset += FLAGS.vis_batch_size
          
      tf.logging.info(
          'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


if __name__ == '__main__':
#  flags.mark_flag_as_required('checkpoint_dir')
#  flags.mark_flag_as_required('vis_logdir')
#  flags.mark_flag_as_required('tfrecord_path')
#  flags.mark_flag_as_required('label_map_path')
#  flags.mark_flag_as_required('list_path')
#  flags.mark_flag_as_required('config_path')
  
  tf.app.run()
