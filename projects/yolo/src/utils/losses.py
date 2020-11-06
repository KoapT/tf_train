#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Editor      : PyCharm
#   File name   : losses.py
#   Author      : Koap
#   Created date: 2020/8/19 下午2:54
#   Description :
#
# ================================================================

from yolo.src.utils.ious import *
import tensorflow as tf

slim = tf.contrib.slim


def focal(target, actual, alpha=1, gamma=2):
    focal_loss = tf.abs(target - alpha) * tf.pow(tf.abs(target - actual), gamma)
    return focal_loss


def cross_entropy(labels, logits):
    return -(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)) + (1 - labels) * tf.log(
        tf.clip_by_value((1 - logits), 1e-10, 1.0)))


def loss_layer_v3(pred, label, bboxes_xywh, input_size,ignore_thresh=0.3,use_focal=False):
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

    giou = tf.expand_dims(bbox_iou(pred_xywh, label_xywh, method='giou'), axis=-1)

    bbox_loss_scale = 1.0 + 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
    # 加入了bbox_loss_scale，尺寸越小物体，bbox_loss_scale越大， giou_loss增大。
    # 通过该手段可以增大小物体对于定位损失的影响，从而使小物体的定位更准确。

    iou = bbox_iou(pred_xywh[:, :, :, :, None, :],
                   bboxes_xywh[:, None, None, None, :, :])
    max_iou = tf.reduce_max(iou, axis=-1, keepdims=True)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < ignore_thresh, tf.float32)
    # 这里的意思是:
    # 对于某个在背景位置的预测框（(1.0 - respond_bbox)），
    # 如果它匹配到了一个gtbox（即max_iou>=ignore_thresh的情况），
    # 则忽略它的loss，（tf.cast(max_iou < ignore_thresh, tf.float32)），
    # 这相当于： 在不该预测到框的anchor位置预测到了一个正确的框。
    # 即：IoU大于阈值，但又不是正样本的achor，不计他们的conf loss

    if use_focal:
        conf_focal = focal(respond_bbox, pred_conf)
    else:
        conf_focal = 1

    conf_loss = conf_focal * (
            respond_bbox * cross_entropy(labels=respond_bbox,
                                         logits=pred_conf)  # 前景（有物体）的置信度损失
            +
            respond_bgd * cross_entropy(labels=respond_bbox,
                                        logits=pred_conf)  # 背景的置信度损失
    )
    prob_loss = respond_bbox * cross_entropy(labels=label_prob, logits=pred_prob)

    # 仅对respond_bbox中=1的部分求prob_loss
    locate_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    locate_loss = tf.check_numerics(locate_loss, 'locate_loss is nan or inf')
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.check_numerics(conf_loss, 'conf_loss is nan or inf')
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))  # 每个batch的所有anchor的loss求和之后，再对batch求平均。
    prob_loss = tf.check_numerics(prob_loss, 'prob_loss is nan or inf')
    tf.summary.scalar('locate_loss', locate_loss)
    tf.summary.scalar('conf_loss', conf_loss)
    tf.summary.scalar('prob_loss', prob_loss)
    return tf.add_n([locate_loss, conf_loss, prob_loss], name='loss_add')


def loss_layer_v4(pred, label, bboxes_xywh, input_size, ignore_thresh=0.4, use_focal=False):
    """
    加入了DIOU 和 CIOU LOSS
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

    ciou = tf.expand_dims(bbox_iou(pred_xywh, label_xywh, method='ciou'), axis=-1)

    bbox_loss_scale = 1.0 + 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)
    # 加入了bbox_loss_scale，尺寸越小物体，bbox_loss_scale越大， ciou_loss增大。
    # 通过该手段可以增大小物体对于定位损失的影响，从而使小物体的定位更准确。

    iou = bbox_iou(pred_xywh[:, :, :, :, None, :],
                   bboxes_xywh[:, None, None, None, :, :],
                   method='diou')
    max_iou = tf.reduce_max(iou, axis=-1, keepdims=True)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < ignore_thresh, tf.float32)
    # 这里的意思是:
    # 对于某个在背景位置的预测框（(1.0 - respond_bbox)），
    # 如果它匹配到了一个gtbox（即max_iou>=ignore_thresh的情况），
    # 则忽略它的loss，（tf.cast(max_iou < ignore_thresh, tf.float32)），
    # 这相当于： 在不该预测到框的anchor位置预测到了一个正确的框。
    # 即：IoU大于阈值，但又不是正样本的achor，不计他们的conf loss

    if use_focal:
        conf_focal = focal(respond_bbox, pred_conf)
    else:
        conf_focal = 1
    conf_loss = conf_focal * (
            respond_bbox * cross_entropy(labels=respond_bbox,
                                         logits=pred_conf)  # 前景（有物体）的置信度损失
            +
            respond_bgd * cross_entropy(labels=respond_bbox,
                                        logits=pred_conf)  # 背景的置信度损失
    )

    prob_loss = respond_bbox * cross_entropy(labels=label_prob, logits=pred_prob)

    # 仅对respond_bbox中=1的部分求prob_loss
    locate_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    locate_loss = tf.check_numerics(locate_loss, 'locate_loss is nan or inf')
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.check_numerics(conf_loss, 'conf_loss is nan or inf')
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))  # 每个batch的所有anchor的loss求和之后，再对batch求平均。
    prob_loss = tf.check_numerics(prob_loss, 'prob_loss is nan or inf')
    tf.summary.scalar('locate_loss', locate_loss)
    tf.summary.scalar('conf_loss', conf_loss)
    tf.summary.scalar('prob_loss', prob_loss)
    return tf.add_n([locate_loss, conf_loss, prob_loss], name='loss_add')
