#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Editor      : PyCharm
#   File name   : preprocess_utils.py
#   Author      : Koap
#   Created date: 2020/8/19 下午3:15
#   Description :
#
# ================================================================

import tensorflow as tf
import numpy as np
import time
from yolo.src.utils.ious import bbox_iou, bbox_iou_np


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
    scale_s, scale_m, scale_l = scales[0], scales[1], scales[2]
    for i in range(scales.shape[0]):
        base_scale_list.append(np.sum(scales[:i] ** 2))

    shape = [np.sum(scales ** 2), anchor_per_scale, 5 + num_classes]
    scales = tf.constant(scales, dtype=tf.float32)  # 即outpusizes
    base_scale = tf.constant(base_scale_list, dtype=tf.float32)

    for n in range(max_num_objects):
        bbox_xywh = bboxes_xywh[n]
        obj_flag = obj_flags[n:n + 1]
        bbox_class_ind = labels[n]

        one_hot = tf.one_hot(bbox_class_ind, num_classes)
        bboxes_xywh_label = tf.concat([bbox_xywh, obj_flag, one_hot], axis=-1)

        bbox_xywh_scaled = 1.0 * bbox_xywh[None, :] / strides[:, None]

        iou = []
        for i in range(3):  # 3种scale
            anchors_xy = tf.floor(bbox_xywh_scaled[i, 0:2]) + .5
            anchors_xy = tf.tile(anchors_xy[None, :], [anchor_per_scale, 1])
            anchors_wh = tf.constant(anchors[i], dtype=tf.float32)
            anchors_xywh = tf.concat([anchors_xy, anchors_wh], axis=-1)

            iou_scale = bbox_iou(bbox_xywh_scaled[i][None, :], anchors_xywh)
            iou.append(iou_scale)

        ious = tf.concat(iou, axis=0)
        best_anchor_ind = tf.where(ious >= .5)[:, 0]
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
    # 如果一个anchor被匹配到了多次，会出现叠加的情况。 还未解决
    # result = tf.where(tf.tile(result[..., 4:5] > 1, [1, 1, 5 + num_classes]), result / 2., result)
    result_s, result_m, result_l = tf.split(result, [scale_s ** 2, scale_m ** 2, scale_l ** 2], axis=0)
    label_sbbox = tf.reshape(result_s, (scale_s, scale_s, anchor_per_scale, 5 + num_classes))
    label_mbbox = tf.reshape(result_m, (scale_m, scale_m, anchor_per_scale, 5 + num_classes))
    label_lbbox = tf.reshape(result_l, (scale_l, scale_l, anchor_per_scale, 5 + num_classes))
    return {'label_sbbox': label_sbbox,
            'label_mbbox': label_mbbox,
            'label_lbbox': label_lbbox}


# 使用pyfunc
def gen_ground_truth_np(bboxes_xywh, labels, num_objs, img_size, strides, anchors, anchor_per_scale, num_classes,
                        max_num_objects):
    """
    处理某张图片中的所有gt_bboxes。
    :return: label_sbbox, label_mbbox, label_lbbox  ->  shape:[ouputsize,outputsize,anchors_per_scale，5+num_classes]
             sbboxes, mbboxes, lbboxes              ->  shape:[num_of_bboxes，4]   4:x,y,w,h
    """
    scales = img_size // strides
    label = [np.zeros((scales[i], scales[i], anchor_per_scale,
                       5 + num_classes), dtype=np.float32) for i in range(3)]
    uniform_distribution = np.full(num_classes, 1.0 / num_classes)  # 均匀分布
    delta = 0.01

    bboxes = [np.zeros((max_num_objects, 4),dtype=np.float32) for _ in range(3)]
    bbox_count = np.zeros((3,))
    if num_objs > 0:
        for obj_i in range(num_objs):
            bbox_xywh = bboxes_xywh[obj_i]
            bbox_class_ind = labels[obj_i]
            onehot = np.zeros(num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution
            # smooth_onehot = onehot

            bbox_xywh_scaled = 1.0 * bbox_xywh[None, :] / strides[:, None]
            iou = []

            exist_positive = False
            for i in range(3):  # 3种scale
                anchors_xywh = np.zeros((anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # anchor 的中心位置设置
                anchors_xywh[:, 2:4] = anchors[i]  # 在ouput_size上，anchor的位置和大小x,y,w,h -> shape:[3,4]
                iou_scale = bbox_iou_np(bbox_xywh_scaled[i][None, :], anchors_xywh,
                                        method='diou')  # 一个box跟对应位置上的3个anchors求iou
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.5
                # print(anchors_xywh)
                # print(bbox_xywh_scaled[i][None, :])
                # print(iou_scale)
                # print("iou_mask",iou_mask)

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)  # 在output map上面的横、纵坐标。

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % max_num_objects)
                    bboxes[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

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

                bbox_ind = int(bbox_count[i] % max_num_objects)
                bboxes[i][bbox_ind, :4] = bbox_xywh
                bbox_count[i] += 1

    label_sbbox, label_mbbox, label_lbbox = label
    sbboxes, mbboxes, lbboxes = bboxes
    return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


def preprocess_true_boxes_pyfunc(bboxes_xywh, labels, num_objs, img_size, strides, anchors, anchor_per_scale,
                                 num_classes,
                                 max_num_objects):
    scales = img_size // strides

    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = tf.py_func(gen_ground_truth_np,
                                                                                  inp=[bboxes_xywh, labels, num_objs,
                                                                                       img_size, strides,
                                                                                       anchors, anchor_per_scale,
                                                                                       num_classes, max_num_objects],
                                                                                  Tout=[tf.float32, tf.float32,
                                                                                        tf.float32, tf.float32,
                                                                                        tf.float32, tf.float32],
                                                                                  name='pyfunc_genGT')
    label_sbbox.set_shape([scales[0], scales[0], anchor_per_scale, 5 + num_classes])
    label_mbbox.set_shape([scales[1], scales[1], anchor_per_scale, 5 + num_classes])
    label_lbbox.set_shape([scales[2], scales[2], anchor_per_scale, 5 + num_classes])
    sbboxes.set_shape([max_num_objects, 4])
    mbboxes.set_shape([max_num_objects, 4])
    mbboxes.set_shape([max_num_objects, 4])

    return {'label_sbbox': label_sbbox,
            'label_mbbox': label_mbbox,
            'label_lbbox': label_lbbox,
            'sbboxes': sbboxes,
            'mbboxes': mbboxes,
            'lbboxes': lbboxes}

## Test preprocess_true_boxes
if __name__ == '__main__':
    bboxes = tf.constant([[0.44192678, 0.82245343, 0.2, 0.1],
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
