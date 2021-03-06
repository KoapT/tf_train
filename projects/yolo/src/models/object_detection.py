#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Editor      : PyCharm
#   File name   : object_detection.py
#   Author      : Koap
#   Created date: 2020/3/11 下午3:44
#   Description :
#
# ================================================================

import tensorflow as tf
import numpy as np
from yolo.src.utils import param_file_access, data_augment, model_utils, losses, \
    shape_utils, visualization_utils, postprocess_utils, preprocess_utils
import yolo.src.datasets.std.inputs_builder as std_inputs_builder
from yolo.src.core import model
from yolo.src.nets import YOLO
import tensorflow.contrib.slim as slim

_NAME_INPUT_IMAGE = 'input_image'
_NAME_OUTPUT_LOGIT = 'output_label'
_SCOPE_PREDICT_LOGIT = 'yolo/', #'neck/'


def _test(sess):
    print('enter test=================')


class ObjectDetection(model.Model):
    def initialize(self, model_info_path, sample_info_path, is_training):
        self._label_map_path = sample_info_path
        self._config_path = model_info_path
        self._config_dict = param_file_access.get_json_params(self._config_path)
        self._label_name_key = self._config_dict['sample_config']['key_name']
        self._build_label_from_name = self._config_dict['sample_config']['build_label_from_name']
        self._batch_norm_params = self._config_dict['model_config']['batch_norm_params']
        self._class_dict = param_file_access.get_label_map_class_id_name_dict(self._label_map_path,
                                                                              key_name=self._label_name_key)  # Yolo的标签从0开始
        print('class dict:', self._class_dict)
        self._num_classes = len(self._class_dict)
        self._category_index = param_file_access.get_category_index(self._label_map_path,
                                                                    key_name=self._label_name_key)
        self._is_training = is_training
        self._batch_size = (self._config_dict['train_config']['batch_size']
                            if is_training
                            else self._config_dict['eval_config']['batch_size'])
        self._input_size = self._config_dict['model_config']['input_size']
        self._max_num_objects_per_image = self._config_dict['sample_config']['max_num_objects_per_image']
        self._is_debug = self._config_dict['model_config']['debug']
        print('Debug: {}'.format(self._is_debug))
        # TODO: add the following items to the config file：
        self._yolo_version = self._config_dict['yolo_version']
        _anchors = self._config_dict['model_config']['anchors_%s' % self._yolo_version]  # scale 之后的anchor
        self._anchor_per_scale = self._config_dict['model_config']['anchor_per_scale']
        self._strides = np.array(self._config_dict['model_config']['strides'])
        self._anchors = np.array(_anchors, dtype=np.float32).reshape([-1, self._anchor_per_scale, 2]) \
                        / self._strides[:, None, None]
        self._visualize_images = self._config_dict['train_config']['visualize_images']
        self._max_num_images_visualized = self._config_dict['train_config']['max_num_images_to_visualize']
        self._weights_decay = self._config_dict['train_config']['weights_regularizer']
        self._num_threads = self._config_dict['train_config']['num_threads']  # 数据预处理的线程数
        if self._yolo_version == 'v3':
            self.net = YOLO.YOLOV3
        elif self._yolo_version == 'v4':
            self.net = YOLO.YOLOV4
        else:
            print('Choose yolo version v3 or v4')
            raise NameError

    def preprocess(self, image, bboxes, fast_mode=True):
        with tf.name_scope('preprocess_image'):
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            num_distort_cases = 1 if fast_mode else 4
            image = data_augment.apply_with_random_selector(
                image,
                lambda x, ordering: data_augment.distort_color(x, ordering, fast_mode),
                num_cases=num_distort_cases)

            # image, bboxes = tf.cond(tf.random_uniform([]) > 0.5,
            #                         lambda: data_augment.random_crop(image, bboxes),
            #                         lambda: data_augment.do_nothing(image, bboxes))
            # image, bboxes = tf.cond(tf.random_uniform([]) > 0.5,
            #                         lambda: data_augment.do_nothing(image, bboxes),
            #                         lambda: data_augment.random_pad_image(image, bboxes))
            image = tf.image.resize_images(
                image, [self._input_size, self._input_size],
                method=tf.image.ResizeMethod.BILINEAR,
                align_corners=True)
            return image, bboxes

    def predict(self, preprocessed_inputs):
        detections = self.net(preprocessed_inputs,
                         num_classes=self._num_classes,
                         batch_norm_params=self._batch_norm_params,
                         activation_fn=tf.nn.leaky_relu,
                         weight_decay=self._weights_decay,
                         is_training=self._is_training)
        return detections

    def loss(self, name, pred, label, bboxes_xywh):
        with tf.name_scope(name):
            if self._yolo_version=='v3':
                loss = losses.loss_layer_v3(pred, label, bboxes_xywh, input_size=self._input_size)
            else:
                loss = losses.loss_layer_v4(pred, label, bboxes_xywh, input_size=self._input_size, use_focal=True)
            return loss

    def postprocess(self, predictions):
        with tf.name_scope('postprocess'):
            predictions_s, detections_s = postprocess_utils.decode(predictions['detect_s'], self._anchors[0],
                                                                   self._num_classes, self._strides[0])
            predictions_m, detections_m = postprocess_utils.decode(predictions['detect_m'], self._anchors[1],
                                                                   self._num_classes, self._strides[1])
            predictions_l, detections_l = postprocess_utils.decode(predictions['detect_l'], self._anchors[2],
                                                                   self._num_classes, self._strides[2])
            detections = tf.concat([detections_s, detections_m, detections_l], axis=1)
            return {'predictions_s': predictions_s,
                    'predictions_m': predictions_m,
                    'predictions_l': predictions_l}, detections

    def provide_groundtruth(self, bboxes_xywh, labels, num_objs, img_size, strides, anchors,
                            anchor_per_scale, num_classes, max_num_objects):
        with tf.name_scope('process_bboxe_to_provided_groundtruths'):
            provided_groundtruths = preprocess_utils.preprocess_true_boxes_pyfunc(bboxes_xywh, labels, num_objs,
                                                                                  img_size,
                                                                                  strides, anchors,
                                                                                  anchor_per_scale, num_classes,
                                                                                  max_num_objects)
            return provided_groundtruths

    def boxes_yxyx2xywh(self, boxes):
        y1, x1, y2, x2 = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        boxes_xyxy = tf.concat([x1, y1, x2, y2], axis=-1)
        boxes_xywh = tf.concat(
            [(boxes_xyxy[..., 2:] + boxes_xyxy[..., :2]) * 0.5, boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]], axis=-1)
        return boxes_xywh

    def build_inputs(self, sample_path, num_samples, batch_size, num_clones):
        builder = std_inputs_builder.StdTFRecordInputs()

        def _build_name_to_label_map(label_map_path, label_name_key):
            name_list = param_file_access.get_label_map_value_list_by_key(label_map_path, key=label_name_key)
            id_list = [i for i in range(0, len(name_list))]
            name_list_tensor = tf.constant(name_list)
            id_list_tensor = tf.constant(id_list, dtype=tf.int64)
            return model_utils.build_tensor_map(
                key_list_tensor=name_list_tensor,
                value_list_tensor=id_list_tensor,
                default_value=-1)

        def preprocess_fn(sample):
            if self._build_label_from_name:
                tensor_map = _build_name_to_label_map(self._label_map_path, self._label_name_key)
                sample[std_inputs_builder.OBJ_LABEL] = model_utils.mapping_tensor(
                    sample[std_inputs_builder.OBJ_LABEL_NAME], tensor_map)
            image = sample[std_inputs_builder.IMAGE]
            num_objs = sample[std_inputs_builder.NUM_OBJS]
            bboxes = sample[std_inputs_builder.OBJ_BOX]
            labels = sample[std_inputs_builder.OBJ_LABEL]
            preprocessed_image, preprocessed_bboxes = self.preprocess(image, bboxes)
            preprocessed_bboxes = preprocessed_bboxes * self._input_size  # change to realsize
            boxes_xywh = self.boxes_yxyx2xywh(preprocessed_bboxes)
            provided_groundtruths = self.provide_groundtruth(boxes_xywh, labels, num_objs,
                                                             img_size=self._input_size,
                                                             strides=self._strides,
                                                             anchors=self._anchors,
                                                             anchor_per_scale=self._anchor_per_scale,
                                                             num_classes=self._num_classes,
                                                             max_num_objects=self._max_num_objects_per_image)

            sample[std_inputs_builder.IMAGE] = preprocessed_image
            sample[std_inputs_builder.OBJ_BOX] = preprocessed_bboxes
            sample['bboxes_xywh'] = boxes_xywh
            sample['label_sbbox'] = provided_groundtruths['label_sbbox']
            sample['label_mbbox'] = provided_groundtruths['label_mbbox']
            sample['label_lbbox'] = provided_groundtruths['label_lbbox']
            return sample

        input_queue = builder.get(
            tfrecord_path=sample_path,
            num_samples=num_samples,
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            num_clones=num_clones,
            num_threads=self._num_threads,
            include_detect=True,
            max_num_labels=self._num_classes,
            max_num_objects=self._max_num_objects_per_image)
        return input_queue

    def build_model(self, input_queue):
        samples = input_queue.dequeue()
        image_names = samples[std_inputs_builder.NAME]
        gt_boxes = samples[std_inputs_builder.OBJ_BOX]
        preprocessed_inputs = samples[std_inputs_builder.IMAGE]
        num_objs = tf.cast(tf.reduce_sum(samples[std_inputs_builder.NUM_OBJS]),tf.float32)
        labels_sbbox = samples['label_sbbox']
        labels_mbbox = samples['label_mbbox']
        labels_lbbox = samples['label_lbbox']
        bboxes_xywh = samples['bboxes_xywh']
        predictions = self.predict(preprocessed_inputs)
        postprocessed_predictions, postprocessed_detections = self.postprocess(predictions)
        loss_s = self.loss('loss_of_sobj', postprocessed_predictions['predictions_s'],
                           labels_sbbox, bboxes_xywh)
        loss_m = self.loss('loss_of_mobj', postprocessed_predictions['predictions_m'],
                           labels_mbbox, bboxes_xywh)
        loss_l = self.loss('loss_of_lobj', postprocessed_predictions['predictions_l'],
                           labels_lbbox, bboxes_xywh)
        loss = loss_s + loss_m + loss_l
        if self._weights_decay > 0.:
            regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            regularization_loss = 0.
        losses = [loss, regularization_loss]
        loss = tf.add_n(losses)

        tf.summary.scalar('Loss/loss_s', loss_s)
        tf.summary.scalar('Loss/loss_m', loss_m)
        tf.summary.scalar('Loss/loss_l', loss_l)
        tf.summary.scalar('Loss/loss', losses[0]/ tf.maximum(1., num_objs)*self._batch_size)
        tf.summary.scalar('Loss/regularization_loss', losses[1])

        postprocessed_detections = tf.identity(postprocessed_detections, name='postprocessed_detections')

        # evaluation metrics
        metric_map = {}

        if self._visualize_images:
            # visualize gt
            images = samples[std_inputs_builder.IMAGE][:self._max_num_images_visualized]
            images_uint8 = tf.image.convert_image_dtype(images, dtype=tf.uint8)
            bboxes = samples[std_inputs_builder.OBJ_BOX][:self._max_num_images_visualized]
            classes = samples[std_inputs_builder.OBJ_LABEL][:self._max_num_images_visualized]
            scores = tf.ones_like(classes, dtype=tf.float32) * tf.cast(classes > -1, dtype=tf.float32)
            image_with_box = visualization_utils.draw_bounding_boxes_on_image_tensors(
                images_uint8, bboxes, classes, scores, self._category_index,
                max_boxes_to_draw=self._max_num_objects_per_image, use_normalized_coordinates=False)
            # visualize predictions
            images2 = model_utils.copytensor(images_uint8)
            bboxes2 = postprocessed_detections[:self._max_num_images_visualized]  # bbox:xywh
            bboxes2_with_score_and_cls = postprocess_utils.nms(bboxes2, confidence_threshold=.3, iou_threshold=.3,
                                                               num_class=self._num_classes)  # bbox: yxyx
            bboxes2_with_score_and_cls = tf.map_fn(
                lambda x: shape_utils.pad_or_clip_nd(tf.boolean_mask(x, x[..., 4] > 0), [20, 6]),
                bboxes2_with_score_and_cls)

            bboxes2 = bboxes2_with_score_and_cls[..., :4]
            scores2 = bboxes2_with_score_and_cls[..., 4]
            classes2 = tf.cast(bboxes2_with_score_and_cls[..., 5], dtype=tf.int64)
            image_with_box2 = visualization_utils.draw_bounding_boxes_on_image_tensors(
                images2, bboxes2, classes2, scores2, self._category_index,
                max_boxes_to_draw=self._max_num_objects_per_image, use_normalized_coordinates=False)

            tf.summary.image('groundtruths', image_with_box,
                             max_outputs=self._max_num_images_visualized)
            tf.summary.image('predictions', image_with_box2,
                             max_outputs=self._max_num_images_visualized)

        train_item = model.TrainItem(
            loss=loss,
            var_list=slim.get_trainable_variables(),  # or =None
            summary_name='')
        train_item_list = [train_item]
        outputs = model.BuildModelOutputs(
            train_item_list=train_item_list,
            metric_dict=metric_map)

        return outputs

    def build_optimizer(self, train_number_of_steps):
        learning_rate = model_utils.get_model_learning_rate(
            learning_policy=self._config_dict['train_config']['learning_policy'],
            base_learning_rate=self._config_dict['train_config']['base_learning_rate'],
            learning_rate_decay_step=self._config_dict['train_config']['learning_rate_decay_step'],
            learning_rate_decay_factor=self._config_dict['train_config']['learning_rate_decay_factor'],
            training_number_of_steps=train_number_of_steps,
            learning_power=self._config_dict['train_config']['learning_power'],
            slow_start_step=self._config_dict['train_config']['slow_start_step'],
            slow_start_learning_rate=self._config_dict['train_config']['slow_start_learning_rate'],
            end_learning_rate=self._config_dict['train_config']['end_learning_rate'])
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # optimizer = tf.train.MomentumOptimizer(
        #     learning_rate=learning_rate,
        #     momentum=self._config_dict['train_config']['momentum'])
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(
        #     learning_rate=learning_rate,
        #     decay=self._config_dict['train_config']['rmsprop_decay'],
        #     momentum=self._config_dict['train_config']['rmsprop_momentum'],
        #     epsilon=self._config_dict['train_config']['opt_epsilon'])

        optimizer_item = model.OptimizerItem(optimizer=optimizer, learn_rate=learning_rate, summary_name='')
        optimizer_item_list = [optimizer_item]
        outputs = model.BuildOptimizerOutputs(optimizer_item_list=optimizer_item_list)

        return outputs

    def schedule_per_train_step(self, train_op_list, step):
        train_op = train_op_list[0]
        return train_op

    def get_batch_size(self):
        return self._batch_size

    def create_for_inference(self):
        inputs = tf.placeholder(tf.uint8, [None, None, None, 3], name=_NAME_INPUT_IMAGE)
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)
        preprocessed_inputs = tf.image.resize_images(
            inputs, [self._input_size, self._input_size],
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=True)
        predictions = self.predict(preprocessed_inputs)
        _, postprocessed_detections = self.postprocess(predictions)
        tf.identity(postprocessed_detections, name=_NAME_OUTPUT_LOGIT)


    def create_for_evaluation(self):
        inputs = tf.placeholder(tf.uint8, [1, self._input_size, self._input_size, 3], name=_NAME_INPUT_IMAGE)
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)
        input_bbox = tf.placeholder(tf.float32, [1, None, 4], name='input_bbox')
        input_box_category = tf.placeholder(tf.int64, [1, None], name='input_box_category')
        input_box_numbers = tf.placeholder(tf.int32, [1], name='input_box_numbers')

        # input_bbox = shape_utils.pad_or_clip_nd(input_bbox,[1,self._max_num_objects_per_image,4])
        # input_box_category = shape_utils.pad_or_clip_nd(input_box_category, [1, self._max_num_objects_per_image])
        # input_box_numbers = shape_utils.pad_or_clip_nd(input_box_numbers, [self._max_num_objects_per_image])

        provided_groundtruths = self.provide_groundtruth(input_bbox[0], input_box_category[0], input_box_numbers[0],
                                                         img_size=self._input_size,
                                                         strides=self._strides,
                                                         anchors=self._anchors,
                                                         anchor_per_scale=self._anchor_per_scale,
                                                         num_classes=self._num_classes,
                                                         max_num_objects=self._max_num_objects_per_image)
        label_sbbox = tf.expand_dims(provided_groundtruths['label_sbbox'],axis=0)
        label_mbbox = tf.expand_dims(provided_groundtruths['label_mbbox'],axis=0)
        label_lbbox = tf.expand_dims(provided_groundtruths['label_lbbox'],axis=0)
        sbboxes = tf.expand_dims(provided_groundtruths['sbboxes'],axis=0)
        mbboxes = tf.expand_dims(provided_groundtruths['mbboxes'], axis=0)
        lbboxes = tf.expand_dims(provided_groundtruths['lbboxes'], axis=0)
        predictions = self.predict(inputs)

        postprocessed_predictions, postprocessed_detections = self.postprocess(predictions)
        loss_s = self.loss('loss_of_sobj', postprocessed_predictions['predictions_s'],
                           label_sbbox, sbboxes)
        loss_m = self.loss('loss_of_mobj', postprocessed_predictions['predictions_m'],
                           label_mbbox, mbboxes)
        loss_l = self.loss('loss_of_lobj', postprocessed_predictions['predictions_l'],
                           label_lbbox, lbboxes)
        loss = loss_s + loss_m + loss_l

        loss_s1 = self.loss('loss_of_sobj1', postprocessed_predictions['predictions_s'],
                           label_sbbox, input_bbox)
        loss_m1 = self.loss('loss_of_mobj1', postprocessed_predictions['predictions_m'],
                           label_mbbox, input_bbox)
        loss_l1 = self.loss('loss_of_lobj1', postprocessed_predictions['predictions_l'],
                           label_lbbox, input_bbox)
        loss1 = loss_s1 + loss_m1 + loss_l1



        tf.identity(loss, name='loss')
        tf.identity(loss1, name='loss1')
        tf.identity(postprocessed_detections, name=_NAME_OUTPUT_LOGIT)

    def get_input_names(self):
        return [_NAME_INPUT_IMAGE]

    def get_output_names(self):
        return [_NAME_OUTPUT_LOGIT]

    def get_extra_layer_scopes(self):
        return [*_SCOPE_PREDICT_LOGIT]

    def inference(self, graph, sess, feeds,
                  model_info_path=None, sample_info_path=None):
        images_np = feeds
        input_images = graph.get_tensor_by_name('%s:0' % _NAME_INPUT_IMAGE)
        output_logits = graph.get_tensor_by_name('%s:0' % _NAME_OUTPUT_LOGIT)
        logits_np = sess.run(output_logits, feed_dict={input_images: images_np})
        logits_np = logits_np.reshape(logits_np.shape[0], -1, logits_np.shape[-1])
        # print(logits_np.shape)
        return logits_np

    def every_before_train_step_callback_fn(self, sess):
        if self._is_debug:
            _test(sess)
        else:
            pass

    def every_after_train_step_callback_fn(self, sess):
        if self._is_debug:
            _test(sess)
        else:
            pass
