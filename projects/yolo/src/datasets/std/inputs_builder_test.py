#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:34:48 2019

@author: rick
"""

import tensorflow as tf

from yolo.src.utils import param_file_access
from yolo.src.datasets.std import inputs_builder
import time
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from yolo.src.utils import model_utils,data_augment,imgaug_utils
import tensorflow.contrib.slim as slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('list_path', '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/Set/train.txt',
                    'List path.')
flags.DEFINE_string('tfrecord_path',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/TFRecord/train.record',
                    'Tfrecord path.')
flags.DEFINE_string('label_map_path',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/LabelMap/label_map.json',
                    'Label map path.')
flags.DEFINE_string('label_name_key', 'display_name',
                    'Label name key in label map, e.g. display_name')
flags.DEFINE_boolean('include_classify', False,
                     'Is include classify.')
flags.DEFINE_boolean('include_detect', True,
                     'Is include detect.')
flags.DEFINE_boolean('include_detect_label', False,
                    'Is include detect label.')
flags.DEFINE_boolean('include_segment', False,
                     'Is include segment.')
flags.DEFINE_integer('resize_height', 512,
                     'Image resized height.')
flags.DEFINE_integer('resize_width', 512,
                     'Image resized width.')
flags.DEFINE_string('image_save_dir', None,
                    'Image save directory')

def _build_name_to_label_map(label_map_path, label_name_key):
    name_list = param_file_access.get_label_map_value_list_by_key(label_map_path, key=label_name_key)
    id_list = [i for i in range(0, len(name_list))]
    name_list_tensor = tf.constant(name_list)
    id_list_tensor = tf.constant(id_list, dtype=tf.int64)
    return model_utils.build_tensor_map(
            key_list_tensor=name_list_tensor,
            value_list_tensor=id_list_tensor,
            default_value=-1)

def _preprocess_fn(sample, fixed_height, fixed_width,fast_mode=False,build_label_from_name=(not FLAGS.include_detect_label)):
    image = sample[inputs_builder.IMAGE]
    num_objs = sample[inputs_builder.NUM_OBJS]
    bboxes = sample[inputs_builder.OBJ_BOX]
    labels = sample[inputs_builder.OBJ_LABEL]
    labels = tf.cast(labels, dtype=tf.int32)

    t0 = time.time()
    # imgaug_utils.PIECEWISEAFFINE['probability'] = .0
    # aug = imgaug_utils.Augment(image, num_objs, bboxes, fixed_height, fixed_width, max_num_objects=100)
    # image, bboxes = aug(p=.5)  # p表示有p的概率会应用数据增强，p取0~1之间。

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # 将unit8的图片类型转成范围在[0,1]的float32类型
    num_distort_cases = 1 if fast_mode else 4
    # image = data_augment.apply_with_random_selector(
    #     image,
    #     lambda x, ordering: data_augment.distort_color(x, ordering, fast_mode),
    #     num_cases=num_distort_cases)

    image, bboxes = tf.cond(tf.random_uniform([]) > .5,
                            lambda: data_augment.random_crop(image, bboxes),
                            lambda: data_augment.do_nothing(image, bboxes))
    image, bboxes = tf.cond(tf.random_uniform([]) > .5,
                            lambda: data_augment.do_nothing(image, bboxes),
                            lambda: data_augment.random_pad_image(image, bboxes))
    new_height = tf.shape(image)[0]
    new_width = tf.shape(image)[1]
    sample[inputs_builder.HEIGHT]=new_height
    sample[inputs_builder.WIDTH]=new_width
    image = tf.image.resize_images(
        image, [fixed_height, fixed_width],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=True)
    image_with_boxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  tf.expand_dims(bboxes, 0))[0]

    print('aug time:', time.time() - t0)

    sample[inputs_builder.IMAGE] = image
    sample[inputs_builder.OBJ_BOX] = bboxes
    # sample[inputs_builder.OBJ_LABEL] = labels
    if inputs_builder.NUM_OBJS in sample and build_label_from_name:
        tensor_map = _build_name_to_label_map(FLAGS.label_map_path, FLAGS.label_name_key)
        sample[inputs_builder.OBJ_LABEL] = model_utils.mapping_tensor(
                sample[inputs_builder.OBJ_LABEL_NAME], tensor_map)

    if inputs_builder.MASK in sample:
        mask = sample[inputs_builder.MASK]
        if mask is not None:
            mask = tf.image.resize_images(
                mask, [fixed_height, fixed_width],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=True)
        sample[inputs_builder.MASK] = mask
    return sample


def _save_image(image, image_format, save_dir, cnt, mask=None, mask_format=None):
    if save_dir is not None:
        print('save image')
        image_save_path = os.path.join(
            save_dir, 'img_{0}.{1}'.format(cnt, image_format))
        cv2.imwrite(image_save_path, image)
        if mask is not None and mask_format is not None:
            mask_save_path = os.path.join(
                save_dir, 'img_{0}_mask.{1}'.format(cnt, mask_format))
            cv2.imwrite(mask_save_path, mask)

        img_np_tmp = np.array(Image.open(image_save_path))
        print(np.shape(img_np_tmp))
        print(img_np_tmp.dtype)
        cnt = cnt + 1
        return cnt


if __name__ == '__main__':
    print('test================================')
    batch_size = 1
    bbox_color_bgr = (0, 255, 0)
    bbox_thickness = 1
    bbox_text_font = cv2.FONT_HERSHEY_SIMPLEX
    bbox_text_scale = 0.4
    bbox_text_thickness = 1
    mask_alpha = 0.6
    max_num_objects = 50
    num_samples = len(param_file_access.get_txt_params(FLAGS.list_path))
    num_classes = len(param_file_access.get_label_map_class_id_list(FLAGS.label_map_path))
    tfrecord_path = FLAGS.tfrecord_path
    label_map_path = FLAGS.label_map_path
    fixed_height = FLAGS.resize_height
    fixed_width = FLAGS.resize_width
    image_save_dir = FLAGS.image_save_dir

    builder = inputs_builder.StdTFRecordInputs()
    with tf.Graph().as_default() as graph:
        preprocess_args = [fixed_height, fixed_width]
        input_queue = builder.get(
            tfrecord_path=tfrecord_path,
            num_samples=num_samples,
            batch_size=batch_size,
            include_classify=FLAGS.include_classify,
            include_detect=FLAGS.include_detect,
            include_segment=FLAGS.include_segment,
            preprocess_fn=_preprocess_fn,
            preprocess_args=preprocess_args,
            max_num_labels=num_classes,
            max_num_objects=max_num_objects,
            is_shuffle=False)
        samples = input_queue.dequeue()

        with tf.Session() as sess:
            tf.tables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            cnt = 0
            over = False
            while not over:
                print('======================')
                samples2 = sess.run(samples)
                name_list = samples2[inputs_builder.NAME]
                image_list = samples2[inputs_builder.IMAGE]
                image_format_list = samples2[inputs_builder.IMAGE_FORMAT]
                for i in range(len(name_list)):
                    print('%d-------------' % i)
                    image = image_list[i]
                    image_format = str(image_format_list[i], 'utf-8')
                    for key, value in samples2.items():
                        v = value[i]
                        t = type(v)
                        if t is bytes:
                            print('{0}: {1}'.format(key, str(v, 'utf-8')))
                        else:
                            if t is np.ndarray:
                                data = v.dtype if np.size(v) > 1000000 else v
                            else:
                                data = v
                            print('{0}: shape={1}, data={2}, dtype={3}'.format(
                                key, np.shape(v), data, v.dtype))

                    image = np.uint8(image * 255)
                    if inputs_builder.MASK in samples2:
                        mask = np.squeeze(samples2[inputs_builder.MASK][i])
                        mask_format = str(samples2[inputs_builder.MASK_FORMAT][i], 'utf-8')
                        mask_image = Image.fromarray(mask)
                        mask_palette = param_file_access.get_label_map_palette(label_map_path)
                        mask_image.putpalette(mask_palette)
                        plt.figure(0)
                        plt.subplot(1, 3, 1)
                        plt.imshow(mask_image)
                        plt.subplot(1, 3, 2)
                        plt.imshow(image)
                        plt.subplot(1, 3, 3)
                        plt.imshow(image)
                        plt.imshow(mask_image, alpha=mask_alpha)
                        plt.show()

                        cnt = _save_image(
                            image, image_format, image_save_dir, cnt,
                            mask, mask_format)
                    else:
                        str_labels = ''
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        if inputs_builder.NUM_OBJS in samples2:
                            num_boxes = samples2[inputs_builder.NUM_OBJS][i]
                            box_label_list = samples2[inputs_builder.OBJ_LABEL][i]
                            box_label_name_list = samples2[inputs_builder.OBJ_LABEL_NAME][i]
                            box_list = samples2[inputs_builder.OBJ_BOX][i]
                            for j in range(num_boxes):
                                box_label = box_label_list[j]
                                box_label_name = str(box_label_name_list[j], 'utf-8')
                                box = box_list[j]
                                box[1] = int(box[1] * fixed_width)  # xmin
                                box[0] = int(box[0] * fixed_height)  # ymin
                                box[3] = int(box[3] * fixed_width)  # xmax
                                box[2] = int(box[2] * fixed_height)  # ymax
                                text = '%s: %d' % (box_label_name, box_label)
                                cv2.rectangle(image,
                                              (box[1], box[0]),
                                              (box[3], box[2]),
                                              bbox_color_bgr,
                                              bbox_thickness)
                                cv2.putText(image,
                                            text,
                                            (box[1], box[0]),
                                            bbox_text_font,
                                            bbox_text_scale,
                                            bbox_color_bgr,
                                            bbox_text_thickness)
                        if inputs_builder.NUM_LABELS in samples2:
                            num_labels = samples2[inputs_builder.NUM_LABELS][i]
                            label_list = samples2[inputs_builder.LABEL][i]
                            str_labels = ''
                            splite_str = ','
                            for j in range(num_labels):
                                id_name_dict = param_file_access.get_label_map_class_id_name_dict(
                                    label_map_path,FLAGS.label_name_key)
                                print(id_name_dict)
                                label_id = label_list[j]
                                str_labels = (
                                        str_labels +
                                        str(label_id) + ':' +
                                        id_name_dict[label_id] + splite_str)
                            str_labels = str_labels.strip(splite_str)
                            print(str_labels)

                        title = str_labels if str_labels is not '' else ''
                        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(title, image)
                        cnt = _save_image(image, image_format, image_save_dir, cnt)
                        if cv2.waitKey(0) & 0xFF == 27:  # enter Esc to quit
                            cv2.destroyAllWindows()
                            over = True
                            break
                        else:  # enter other key to next
                            cv2.destroyAllWindows()
                            continue

            cv2.destroyAllWindows()

            coord.request_stop()
            coord.join(threads)