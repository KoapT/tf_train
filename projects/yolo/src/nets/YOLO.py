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
from yolo.src.nets.blocks import conv, darknet53, yolo_block, upsample_block, cspdarknet53, PANneck

def YOLOV3(inputs, num_classes, batch_norm_params, activation_fn, weight_decay, is_training=True):
    batch_norm_params['is_training'] = is_training

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: activation_fn(x, alpha=.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with tf.variable_scope('darknet-53'):
            route_1, route_2, inputs = darknet53(inputs)

        with tf.variable_scope('yolo'):
            route, inputs = yolo_block(inputs, 512)
            detect_l = slim.conv2d(inputs, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='Conv_detect_l')
            # 为了方便从darknet的权重转换过来，卷积层的命名中要含有'Conv'字符，
            # 因为slim.conv2d的默认命名是Conv_i，下同。

            inputs = conv(route, 256, 1)
            upsample_shape = tf.shape(route_2)
            inputs = upsample_block(inputs, upsample_shape)
            inputs = tf.concat([inputs, route_2],
                               axis=-1)

            route, inputs = yolo_block(inputs, 256)
            detect_m = slim.conv2d(inputs, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='Conv_detect_m')

            inputs = conv(route, 128, 1)
            upsample_shape = tf.shape(route_1)
            inputs = upsample_block(inputs, upsample_shape)
            inputs = tf.concat([inputs, route_1],
                               axis=-1)

            _, inputs = yolo_block(inputs, 128)
            detect_s = slim.conv2d(inputs, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='Conv_detect_s')

    return {'detect_s': detect_s, 'detect_m': detect_m, 'detect_l': detect_l}


def YOLOV4(inputs, num_classes, batch_norm_params, activation_fn, weight_decay=0.0005, is_training=True):
    # set batch norm params
    batch_norm_params['is_training'] = is_training

    up_route_1, up_route_2, net = cspdarknet53(inputs, batch_norm_params, weight_decay)
    route_1, route_2, route_3 = PANneck(net, up_route_1, up_route_2,
                                        batch_norm_params, weight_decay)

    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: activation_fn(x, alpha=.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with tf.variable_scope('yolo'):
            route, net = yolo_block(route_1, 128)
            detect_s = slim.conv2d(net, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='Conv_detect_s')

            net = conv(route, 256, stride=2)
            net = tf.concat([net, route_2], -1)
            route, net = yolo_block(net, 256)
            detect_m = slim.conv2d(net, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='Conv_detect_m')
            net = conv(route, 512, stride=2)
            net = tf.concat([net, route_3], -1)
            _, net = yolo_block(net, 512)
            detect_l = slim.conv2d(net, 3 * (5 + num_classes), 1,
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None,
                                   biases_initializer=tf.zeros_initializer(),
                                   scope='Conv_detect_l')

    return {'detect_s': detect_s, 'detect_m': detect_m, 'detect_l': detect_l}
