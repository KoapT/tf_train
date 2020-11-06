#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Editor      : PyCharm
#   File name   : darknet53.py
#   Author      : Koap
#   Created date: 2020/3/11 下午5:33
#   Description :
#
# ================================================================
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time


# tf.enable_eager_execution()


def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)
    inputs = _residual_block(inputs, 32)
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _residual_block(inputs, 64)

    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _residual_block(inputs, 128)

    route_1 = inputs
    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)

    for i in range(8):
        inputs = _residual_block(inputs, 256)

    route_2 = inputs
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)

    for i in range(4):
        inputs = _residual_block(inputs, 512)

    return route_1, route_2, inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _residual_block(inputs, filters):
    shortcut = inputs
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)

    inputs = tf.add(inputs, shortcut)
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, mode='CONSTANT'):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def _yolo_block(inputs, filters):
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    route = inputs
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    return route, inputs


def _get_size(img_tensor):
    return img_tensor.get_shape().as_list()[1:3]


def decode(conv_output, anchors, num_class, stride):
    """
    :return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
           contains (x, y, w, h, score, probability)
    """

    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_per_scale = len(anchors)

    conv_output = tf.reshape(conv_output,
                             [batch_size, output_size, output_size, anchor_per_scale, 5 + num_class])

    conv_raw_dxdy = conv_output[..., 0:2]
    conv_raw_dwdh = conv_output[..., 2:4]
    conv_raw_conf = conv_output[..., 4:5]
    conv_raw_prob = conv_output[..., 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, None], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[None, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, None], y[:, :, None]], axis=-1)
    xy_grid = tf.tile(xy_grid[None, :, :, None, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    result = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    detections = tf.reshape(result, (-1, output_size * output_size * anchor_per_scale, 5 + num_class))

    return result, detections


def _upsample(inputs, out_shape):
    new_height = out_shape[1]
    new_width = out_shape[2]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
    inputs = tf.identity(inputs, name='upsampled')
    return inputs


def YOLOV3(inputs, num_classes, batch_norm_params, activation_fn, weight_decay, is_training=True):
    img_size = _get_size(inputs)
    batch_norm_params['is_training'] = is_training

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: activation_fn(x, alpha=.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with tf.variable_scope('darknet-53'):
            route_1, route_2, inputs = darknet53(inputs)

        with tf.variable_scope('deepin-v3'):
            route, inputs = _yolo_block(inputs, 512)
            detect_l = slim.conv2d(inputs, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='detect_l')

            inputs = _conv2d_fixed_padding(route, 256, 1)
            upsample_size = route_2.get_shape().as_list()
            inputs = _upsample(inputs, upsample_size)
            inputs = tf.concat([inputs, route_2],
                               axis=3)

            route, inputs = _yolo_block(inputs, 256)
            detect_m = slim.conv2d(inputs, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='detect_m')

            inputs = _conv2d_fixed_padding(route, 128, 1)
            upsample_size = route_1.get_shape().as_list()
            inputs = _upsample(inputs, upsample_size)
            inputs = tf.concat([inputs, route_1],
                               axis=3)

            _, inputs = _yolo_block(inputs, 128)
            detect_s = slim.conv2d(inputs, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='detect_s')

            return {'detect_s': detect_s, 'detect_m': detect_m, 'detect_l': detect_l}


# # 不使用py_func
def preprocess_true_boxes(bboxes_xywh, labels, num_objs, img_size, strides, anchors, anchor_per_scale, num_classes,
                          max_num_objects):
    '''
    :param bboxes_xywh: tensor, [max_num_of_bboxes, 4]->xywh
    :param labels: tensor, [max_num_of_bboxes,]
    :param anchors: array [scalse,anchor_per_scale,2]
    :param anchor_per_scale: default 3
    :param scales: [52,26,13] , list
    :return:
    '''

    obj_flags = tf.pad(tf.ones(num_objs), [[0, max_num_objects - num_objs]])
    indice_list = []
    update_list = []
    base_scale_list = []
    scales = img_size // strides
    for i in range(scales.shape[0]):
        base_scale_list.append(np.sum(scales[:i] ** 2))

    shape = [np.sum(scales ** 2), anchor_per_scale, 5 + num_classes]
    scales = tf.constant(scales, dtype=tf.float32)  # 即outpusizes
    base_scale = tf.constant(base_scale_list, dtype=tf.float32)

    for n in range(max_num_objects)[:2]:
        bbox_xywh = bboxes_xywh[n]
        obj_flag = obj_flags[n:n + 1]
        bbox_class_ind = labels[n]

        one_hot = tf.one_hot(bbox_class_ind, num_classes)
        bboxes_xywh_label = tf.concat([bbox_xywh, obj_flag, one_hot], axis=-1)

        bbox_xywh_scaled = bbox_xywh[None, :] * scales[:, None]

        iou = []
        for i in range(3):  # 3种scale
            anchors_xy = tf.floor(bbox_xywh_scaled[i, 0:2]) + .5
            anchors_xy = tf.tile(anchors_xy[None, :], [anchor_per_scale, 1])
            anchors_wh = tf.constant(anchors[i], dtype=tf.float32)
            anchors_xywh = tf.concat([anchors_xy, anchors_wh], axis=-1)

            iou_scale = bbox_iou(bbox_xywh_scaled[i][None, :], anchors_xywh)
            iou.append(iou_scale)

        ious = tf.concat(iou, axis=0)
        best_anchor_ind = tf.where(ious >= .9)[:, 0]
        best_anchor_ind = tf.cond(tf.shape(best_anchor_ind)[0] > 0,
                                  lambda: best_anchor_ind,
                                  lambda: tf.argmax(ious[:, None], axis=0))
        best_detect = best_anchor_ind // anchor_per_scale  # scale编号
        best_anchor = best_anchor_ind % anchor_per_scale  # anchor编号

        y_ind = tf.floor(bbox_xywh_scaled[:, 1])
        x_ind = tf.floor(bbox_xywh_scaled[:, 0])
        xy_ind = y_ind * scales + x_ind
        det_xy_ind = tf.gather(xy_ind[None, :], best_detect, axis=1)
        scale_to_add = tf.gather(base_scale[None, :], best_detect, axis=1)
        det_xy_ind = tf.cast((scale_to_add + det_xy_ind)[0], dtype=tf.int64)

        indice = tf.stack([det_xy_ind, best_anchor], axis=-1)
        update = tf.tile(bboxes_xywh_label[None, :], [tf.shape(indice)[0], 1])

        indice_list.append(indice)
        update_list.append(update)

    indices = tf.concat(indice_list, axis=0)
    updates = tf.concat(update_list, axis=0)

    result = tf.scatter_nd(indices, updates, shape)
    # result = tf.where(tf.tile(result[...,4:5]>1,[1,1,5 + num_classes]),result/2.,result)
    # result
    # result_s, result_m, result_l = tf.split(result, [scale_s ** 2, scale_m ** 2, scale_l ** 2], axis=0)
    # label_sbbox = tf.reshape(result_s, (scale_s, scale_s, anchor_per_scale, 5 + num_classes))
    # label_mbbox = tf.reshape(result_m, (scale_m, scale_m, anchor_per_scale, 5 + num_classes))
    # label_lbbox = tf.reshape(result_l, (scale_l, scale_l, anchor_per_scale, 5 + num_classes))
    return result


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / (union_area + 1e-10)


# 使用pyfunc
def preprocess_true_boxes_pyfunc(bboxes_xywh, labels, num_objs, img_size, strides, anchors, anchor_per_scale,
                                 num_classes,
                                 max_num_objects):
    scales = img_size // strides

    def deal_with_np(bboxes_xywh, labels, num_objs):
        """
        处理某张图片中的所有gt_bboxes。
        :return: label_sbbox, label_mbbox, label_lbbox  ->  shape:[ouputsize,outputsize,anchors_per_scale，5+num_classes]
                 sbboxes, mbboxes, lbboxes              ->  shape:[num_of_bboxes，4]   4:x,y,w,h
        """
        label = [np.zeros((scales[i], scales[i], anchor_per_scale,
                           5 + num_classes), dtype=np.float32) for i in range(3)]
        uniform_distribution = np.full(num_classes, 1.0 / num_classes)  # 均匀分布
        delta = 0.01

        if num_objs > 0:
            for obj_i in range(num_objs):
                bbox_xywh = bboxes_xywh[obj_i]
                bbox_class_ind = labels[obj_i]
                onehot = np.zeros(num_classes, dtype=np.float)
                onehot[bbox_class_ind] = 1.0
                smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution

                bbox_xywh_scaled = 1.0 * bbox_xywh[None, :] / strides[:, None]
                iou = []
                exist_positive = False
                for i in range(3):  # 3种scale
                    anchors_xywh = np.zeros((anchor_per_scale, 4))
                    anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # anchor 的中心位置设置
                    anchors_xywh[:, 2:4] = anchors[i]  # 在ouput_size上，anchor的位置和大小x,y,w,h -> shape:[3,4]

                    iou_scale = bbox_iou_np(bbox_xywh_scaled[i][None, :], anchors_xywh)  # 一个box跟对应位置上的3个anchors求iou
                    iou.append(iou_scale)
                    iou_mask = iou_scale > 0.333
                    # print("iou_mask",iou_mask)

                    if np.any(iou_mask):
                        xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)  # 在output map上面的横、纵坐标。

                        label[i][yind, xind, iou_mask, :] = 0
                        label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                        label[i][yind, xind, iou_mask, 4:5] = 1.0
                        label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                        exist_positive = True

                if not exist_positive:  # 如果bbox与三种scale的每个anchor都没有IOU>0.3的，选IOU最大的进行匹配。
                    best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                    best_detect = int(best_anchor_ind / anchor_per_scale)  # scale编号
                    best_anchor = int(best_anchor_ind % anchor_per_scale)  # anchor编号
                    xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                    label[best_detect][yind, xind, best_anchor, :] = 0
                    label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                    label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                    label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

        label_sbbox, label_mbbox, label_lbbox = label
        return label_sbbox, label_mbbox, label_lbbox

    label_sbbox, label_mbbox, label_lbbox = tf.py_func(deal_with_np, [bboxes_xywh, labels, num_objs],
                                                       [tf.float32, tf.float32, tf.float32])
    label_sbbox.set_shape([scales[0], scales[0], anchor_per_scale, 5 + num_classes])
    label_mbbox.set_shape([scales[1], scales[1], anchor_per_scale, 5 + num_classes])
    label_lbbox.set_shape([scales[2], scales[2], anchor_per_scale, 5 + num_classes])

    return {'label_sbbox': label_sbbox,
            'label_mbbox': label_mbbox,
            'label_lbbox': label_lbbox}


def bbox_iou_np(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / (union_area + 1e-10)


def bbox_giou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-10)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-10)
    return giou


def focal(target, actual, alpha=1, gamma=2):
    focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
    return focal_loss


def cross_entropy(labels, logits):
    return -(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)) + (1 - labels) * tf.log(
        tf.clip_by_value((1 - logits), 1e-10, 1.0)))


def loss_layer(pred, label, bboxes_xywh, input_size):
    """
    pred: 经过decode之后的结果：[batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]， 其中最后一维的x,y,w,h是在input size上的。
    label: shape同pred,每个pred对应一个label。
    bboxes_xywh: [batchsize,num_per_img,xywh]
    """
    input_size = tf.cast(input_size, tf.float32)

    pred_xywh = pred[..., 0:4]
    pred_conf = pred[..., 4:5]
    pred_prob = pred[..., 5:]

    label_xywh = label[..., 0:4]
    respond_bbox = label[..., 4:5]
    label_prob = label[..., 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    giou = tf.clip_by_value(giou, 0, 1)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
    # 加入了bbox_loss_scale，尺寸越小物体，bbox_loss_scale越大， giou_loss增大。
    # 通过该手段可以增大小物体对于定位损失的影响，从而使小物体的定位更准确。

    iou = bbox_iou(pred_xywh[:, :, :, :, None, :], bboxes_xywh[:, None, None, None, :, :])
    max_iou = tf.reduce_max(iou, axis=-1, keepdims=True)  # 这里IOU可以用DIOU替换
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < .333, tf.float32)
    # 这里的意思是:
    # 对于某个在背景位置的预测框（(1.0 - respond_bbox)），
    # 如果它匹配到了一个gtbox（即max_iou>self.iou_loss_thresh的情况），
    # 则忽略它的loss，（tf.cast(max_iou < self.iou_loss_thresh, tf.float32)），
    # 这相当于： 在不该预测到框的anchor位置预测到了一个正确的框。
    # 即：IoU大于阈值，但又不是正样本的achor，不计他们的conf loss

    conf_focal = focal(respond_bbox, pred_conf)
    # conf_loss = conf_focal * (
    #         respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
    #                                                                logits=conv_raw_conf)  # 前景（有物体）的置信度损失
    #         +
    #         respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
    #                                                               logits=conv_raw_conf)  # 背景的置信度损失
    # )
    # prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob,
    #                                                                    logits=conv_raw_prob)

    conf_loss = conf_focal * (
            respond_bbox * cross_entropy(labels=respond_bbox,
                                         logits=pred_conf)  # 前景（有物体）的置信度损失
            +
            respond_bgd * cross_entropy(labels=respond_bbox,
                                        logits=pred_conf)  # 背景的置信度损失
    )
    prob_loss = respond_bbox * cross_entropy(labels=label_prob, logits=pred_prob)

    # 仅对respond_bbox中=1的部分求prob_loss
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    giou_loss = tf.check_numerics(giou_loss, 'giou_loss is nan or inf')
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.check_numerics(conf_loss, 'conf_loss is nan or inf')
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))  # 每个batch的所有anchor的loss求和之后，再对batch求平均。
    prob_loss = tf.check_numerics(prob_loss, 'prob_loss is nan or inf')
    tf.summary.scalar('giou_loss', giou_loss)
    tf.summary.scalar('conf_loss', conf_loss)
    tf.summary.scalar('prob_loss', prob_loss)
    return tf.add_n([giou_loss, conf_loss, prob_loss], name='loss_add')


def nms(batched_predictions, iou_threshold, confidence_threshold, num_class, max_num_obj_per_class=20):
    num_batch = batched_predictions.shape.as_list()[0]
    results = []
    for i in range(num_batch):
        result = []
        predictions = batched_predictions[i]
        predictions = tf.boolean_mask(predictions, predictions[..., 4] > confidence_threshold)
        box_attr = predictions[..., :5]
        box_cls = predictions[..., 5:]
        box_cls = tf.cast(tf.argmax(box_cls, axis=-1), dtype=tf.float32)
        for cls in range(num_class):
            cls_mask = tf.logical_and(box_cls > cls - 1, box_cls < cls + 1)
            cls_pred = tf.boolean_mask(box_attr, cls_mask)
            cls_box_cls = tf.boolean_mask(box_cls[:, None], cls_mask)
            if cls_pred.shape.as_list()[0] == 0:
                result.append(tf.zeros([max_num_obj_per_class, 6]))
            else:
                boxes = cls_pred[..., :4]
                scores = cls_pred[..., 4:5]
                boxes_yxyx = boxes_xywh2yxyx(boxes)
                boxes_with_score_and_cls = tf.concat([boxes_yxyx, scores, cls_box_cls], axis=-1)
                selected_indices = tf.image.non_max_suppression(boxes_yxyx, scores[:, 0], max_num_obj_per_class,
                                                                iou_threshold)
                selected_boxes = tf.gather(boxes_with_score_and_cls, selected_indices)
                number = tf.shape(selected_boxes)[0]
                selected_boxes = tf.cond(tf.less(number, max_num_obj_per_class),
                                         lambda: tf.concat([selected_boxes,
                                                            tf.zeros([max_num_obj_per_class - number, 6])], axis=0),
                                         lambda: selected_boxes[:max_num_obj_per_class, :]
                                         )
                result.append(selected_boxes)
        result_tensor = tf.concat(result, axis=0)
        results.append(result_tensor)
    return tf.stack(results, axis=0)


def boxes_xywh2yxyx(boxes):
    x, y, w, h = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    boxes_yxyx = tf.concat(
        [y - h / 2., x - w / 2., y + h / 2., x + w / 2.], axis=-1)
    return boxes_yxyx


## Test preprocess_true_boxes
if __name__ == '__main__':
    bboxes = tf.constant([[0.74192678, 0.82245343, 0.2, 0.1],
                          [0.718381, 0.8899331, 0.2, 0.1],
                          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]
                          ]) * 8
    labels = tf.constant([1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1], dtype=tf.int64)
    num_objs = tf.constant(2, dtype=tf.int32)
    img_size = 8
    strides = np.array([2, 4, 8])
    anchors = np.array(
        [1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375, 3.625, 2.8125, 4.875,
         6.1875, 11.65625, 10.1875]).reshape([3, 3, 2])
    anchor_per_scale = 3
    num_classes = 3
    max_num_objects = 20
    t0 = time.time()
    label = preprocess_true_boxes_pyfunc(bboxes, labels, num_objs, img_size, strides, anchors, anchor_per_scale,
                                         num_classes,
                                         max_num_objects)
    # print(label)
    # print(label.shape)
    # print('time cost:', time.time() - t0)

    # label = label[None,...]
    # label_xywh = label[..., 0:4]
    # respond_bbox = label[..., 4:5]
    # label_prob = label[..., 5:]
    # bboxes = tf.boolean_mask(label_xywh, tf.cast(respond_bbox[..., 0], 'bool'))
    # pre = tf.ones_like(label_xywh)
    # ious = bbox_iou(label_xywh,bboxes)

    # batched_predictions = tf.constant([[[0.74192678, 0.82245343, 0.2, 0.1, 1, .1, .2, .8],
    #                                     [0.74192678, 0.82245343, 0.2, 0.1, .8, .1, .2, .8]],
    #                                    [[0.74192678, 0.82245343, 0.2, 0.1, 1, .1, .2, .8],
    #                                     [0.74192678, 0.82245343, 0.2, 0.1, 1, 1., .9, .8]]])
    # result = nms(batched_predictions, 0.5, 0.5, 3, max_num_obj_per_class=20)
    sess = tf.Session()
    print(sess.run(label))
