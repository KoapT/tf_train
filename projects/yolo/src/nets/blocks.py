#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 08:58:36 2019

@author: rick
"""

import sys
import tensorflow as tf
from tensorflow.python.ops import init_ops

slim = tf.contrib.slim


# conv2d
def conv(inputs, out_channels, kernel_size=3, stride=1):
    '''
    inputs: tensor
    out_channels: output channels  int
    kernel_size: kernel size int
    stride: int
    return:tensor
    ...
    conv2d:
        input : [batch, height, width, channel]
        kernel : [height, width, in_channels, out_channels]
    '''
    # fixed edge of tensor
    if stride > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    #
    inputs = slim.conv2d(inputs, out_channels, kernel_size, stride=stride,
                         padding=('SAME' if stride == 1 else 'VALID'))
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, mode='CONSTANT'):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def upsample_block(inputs, out_shape):
    new_height = out_shape[1]
    new_width = out_shape[2]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
    inputs = tf.identity(inputs, name='upsampled')
    return inputs


def mish(inputs):
    return inputs * tf.tanh(tf.nn.softplus(inputs))


def _residual_block(inputs, filters, filters_out=None):
    if not filters_out:
        filters_out = filters * 2
    shortcut = inputs
    inputs = conv(inputs, filters, 1)
    inputs = conv(inputs, filters_out)

    inputs = tf.add(inputs, shortcut)
    return inputs


# V4
# implement residual block of yolov4
# : conv(down) + CSPn + conv
def yolo_res_block(inputs, in_channels, res_num, double_ch=False):
    '''
    implement residual block of yolov4
    inputs: tensor
    res_num: run res_num  residual block
    '''
    out_channels = in_channels
    if double_ch:
        out_channels = in_channels * 2

    net = conv(inputs, in_channels * 2, stride=2)
    route = conv(net, out_channels, kernel_size=1)  # in_channels
    net = conv(net, out_channels, kernel_size=1)  # in_channels

    for _ in range(res_num):
        net = _residual_block(net, in_channels, out_channels)

    net = conv(net, out_channels, kernel_size=1)  # in_channels
    net = tf.concat([net, route], -1)
    net = conv(net, in_channels * 2, kernel_size=1)
    return net


# V4
# conv block that kernel is 3*3 and 1*1
def yolo_conv_block(net, in_channels, a, b):
    '''
    net: tensor
    a: the number of conv is a and the kernel size is interleaved 1*1 and 3*3
    b: number of 1*1 convolution
    '''
    for _ in range(a):
        out_channels = in_channels / 2
        net = conv(net, out_channels, kernel_size=1)
        net = conv(net, in_channels)

    out_channels = in_channels
    for _ in range(b):
        out_channels = out_channels / 2
        net = conv(net, out_channels, kernel_size=1)

    return net


def yolo_block(inputs, filters):
    inputs = conv(inputs, filters, 1)
    inputs = conv(inputs, filters * 2)
    inputs = conv(inputs, filters, 1)
    inputs = conv(inputs, filters * 2)
    inputs = conv(inputs, filters, 1)
    route = inputs
    inputs = conv(inputs, filters * 2)
    return route, inputs


# spp maxpool block
def yolo_spp_block(inputs):
    '''
    spp
    inputs:[N, 19, 19, 512]
    return:[N, 19, 19, 2048]
    '''
    max_5 = tf.nn.max_pool(inputs, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
    max_9 = tf.nn.max_pool(inputs, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
    max_13 = tf.nn.max_pool(inputs, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
    # concat
    net = tf.concat([max_13, max_9, max_5, inputs], -1)
    return net


def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    inputs = conv(inputs, 32)
    inputs = conv(inputs, 64, stride=2)
    inputs = _residual_block(inputs, 32)
    inputs = conv(inputs, 128, stride=2)

    for i in range(2):
        inputs = _residual_block(inputs, 64)

    inputs = conv(inputs, 256, stride=2)

    for i in range(8):
        inputs = _residual_block(inputs, 128)

    route_1 = inputs
    inputs = conv(inputs, 512, stride=2)

    for i in range(8):
        inputs = _residual_block(inputs, 256)

    route_2 = inputs
    inputs = conv(inputs, 1024, stride=2)

    for i in range(4):
        inputs = _residual_block(inputs, 512)

    return route_1, route_2, inputs


# V4
def cspdarknet53(inputs, batch_norm_params, weight_decay):
    '''
    Backbone of yolov4
    inputs:[N, 608, 608, 3]
    '''
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: mish(x),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with tf.variable_scope('cspdarknet53'):
            net = conv(inputs, 32)
            net = yolo_res_block(net, 32, 1, double_ch=True)  # *2
            net = yolo_res_block(net, 64, 2)
            net = yolo_res_block(net, 128, 8)
            up_route_54 = net
            net = yolo_res_block(net, 256, 8)
            up_route_85 = net
            net = yolo_res_block(net, 512, 4)
    return up_route_54, up_route_85, net


# V4
def PANneck(inputs, up_route_1, up_route_2, batch_norm_params, weight_decay):
    '''
    Neck of yolov4
    '''
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with tf.variable_scope('neck'):
            net = yolo_conv_block(inputs, 1024, 1, 1)
            net = yolo_spp_block(net)
            net = conv(net, 512, kernel_size=1)
            net = conv(net, 1024)
            net = conv(net, 512, kernel_size=1)
            route_3 = net
            net = conv(net, 256, kernel_size=1)

            up_route_2 = conv(up_route_2, 256, kernel_size=1)
            upsample_shape = tf.shape(up_route_2)
            net = upsample_block(net, upsample_shape)
            net = tf.concat([up_route_2, net], axis=-1)

            net = yolo_conv_block(net, 512, 2, 1)
            route_2 = net
            net = conv(net, 128, kernel_size=1)

            up_route_1 = conv(up_route_1, 128, kernel_size=1)
            upsample_shape = tf.shape(up_route_1)
            net = upsample_block(net, upsample_shape)
            route_1 = tf.concat([up_route_1, net], axis=-1)

    return route_1, route_2, route_3


@slim.add_arg_scope
def bottleneckblock(inputs, num_outputs_neck, num_outputs, is_separable=False,
                    skip_last_activation=False,
                    skip_last_normalizer=False,
                    skip_last_biases=False,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    scope=None):
    default_scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
        inputs = tf.identity(inputs, name='identity_input')
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            biases_initializer=biases_initializer):
            net = slim.conv2d(inputs, kernel_size=1, num_outputs=num_outputs_neck, scope='conv2d')
            if is_separable:
                net = slim.separable_conv2d(net, kernel_size=3, depth_multiplier=1,
                                            num_outputs=None, scope='dwconv2d')
                net = slim.conv2d(net, kernel_size=1, num_outputs=num_outputs,
                                  activation_fn=None if skip_last_activation else activation_fn,
                                  normalizer_fn=None if skip_last_normalizer else normalizer_fn,
                                  biases_initializer=None if skip_last_biases else biases_initializer,
                                  scope='conv2d_1')
            else:
                net = slim.conv2d(net, kernel_size=3, num_outputs=num_outputs,
                                  activation_fn=None if skip_last_activation else activation_fn,
                                  normalizer_fn=None if skip_last_normalizer else normalizer_fn,
                                  biases_initializer=None if skip_last_biases else biases_initializer,
                                  scope='conv2d_2')
        outputs = tf.identity(net, name='identity_output')
    return outputs


@slim.add_arg_scope
def res50block(inputs, compress_rate=2, skip_last_biases=True,
               activation_fn=tf.nn.relu,
               normalizer_fn=None,
               normalizer_params=None,
               biases_initializer=init_ops.zeros_initializer(),
               scope=None):
    default_scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
        inputs = tf.identity(inputs, name='identity_input')
        input_channels = inputs.get_shape().as_list()[3]
        channels = int(input_channels / compress_rate)

        net = inputs
        # pre-activate
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            normalizer_params['scope'] = 'bn'
            net = normalizer_fn(net, **normalizer_params)
        if activation_fn is not None:
            net = activation_fn(net)

        with slim.arg_scope([slim.conv2d],
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            biases_initializer=biases_initializer):
            net = slim.conv2d(net, kernel_size=1, num_outputs=channels, scope='conv2d')
            net = slim.conv2d(net, kernel_size=3, num_outputs=input_channels,
                              activation_fn=None,
                              normalizer_fn=None,
                              biases_initializer=None if skip_last_biases else biases_initializer,
                              scope='conv2d_1')
        net = tf.add(net, inputs, name='add')
        outputs = tf.identity(net, name='identity_output')
    return outputs


@slim.add_arg_scope
def res101block(inputs, compress_rate=2, is_separable=False, skip_last_biases=True,
                activation_fn=tf.nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                biases_initializer=init_ops.zeros_initializer(),
                scope=None):
    default_scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
        inputs = tf.identity(inputs, name='identity_input')
        input_channels = inputs.get_shape().as_list()[3]
        channels = int(input_channels / compress_rate)

        net = inputs
        # pre-activate
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            normalizer_params['scope'] = 'bn'
            net = normalizer_fn(net, **normalizer_params)
        if activation_fn is not None:
            net = activation_fn(net)

        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            biases_initializer=biases_initializer):
            net = slim.conv2d(net, kernel_size=1, num_outputs=channels, scope='conv2d')
            if is_separable:
                net = slim.separable_conv2d(
                    net,
                    kernel_size=3,
                    depth_multiplier=1,
                    num_outputs=None,
                    scope='dwconv2d')
                net = slim.conv2d(net, kernel_size=1, num_outputs=input_channels,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=None if skip_last_biases else biases_initializer,
                                  scope='conv2d_1')
            else:
                net = slim.conv2d(net, kernel_size=3, num_outputs=channels, scope='conv2d_2')
                net = slim.conv2d(net, kernel_size=1, num_outputs=input_channels,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=None if skip_last_biases else biases_initializer,
                                  scope='conv2d_3')
        net = tf.add(net, inputs, name='add')
        outputs = tf.identity(net, name='identity_output')
    return outputs


def denseblock(inputs, level=4, growth_rate=8, is_tiny=True, is_separable=False, scope=None):
    default_scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
        inputs = tf.identity(inputs, name='identity_input')
        for i in range(level):
            fix_str = '' if i == 0 else '_{}'.format(i)
            if is_tiny:
                net = slim.conv2d(inputs, kernel_size=1,
                                  num_outputs=growth_rate,
                                  scope='conv2d{}'.format(fix_str))
                if is_separable:
                    net = slim.separable_conv2d(net, kernel_size=3,
                                                num_outputs=growth_rate,
                                                depth_multiplier=1,
                                                scope='separable_conv2d{}'.format(fix_str))
                else:
                    net = slim.conv2d(net, kernel_size=3,
                                      num_outputs=growth_rate,
                                      scope='conv2d{}_'.format(fix_str))
            else:
                net = slim.conv2d(inputs, kernel_size=3,
                                  num_outputs=growth_rate,
                                  scope='conv2d{}'.format(fix_str))
            inputs = tf.concat([net, inputs], axis=3)
        outputs = tf.identity(inputs, name='identity_output')
    return outputs


def asppblock(inputs, atrous_rate_list, num_outputs_list, scope=None):
    '''atrous_rate_list: atrous rate list, e.g. [0,1,2,3] or
        [[0,0],[1,2],[2,2],[3,2]] as [[height wise rate, width wise rate],...],
        if the list contain 0 or [0,_] or [_,0], then add a conv1*1 route,
        otherwise add a conv3*3 route.
       num_outputs_list: number of output channels per atrous according to atrous_rate_list,
        e.g. [32,32,32,64].
    '''
    default_scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
        inputs = tf.identity(inputs, name='identity_input')
        routes = []
        for i in range(len(atrous_rate_list)):
            rate = atrous_rate_list[i]
            channels = num_outputs_list[i]
            scope = 'conv2d' if i == 0 else 'conv2d_%d' % (i)
            if (isinstance(rate, list) and 0 in rate) or (rate == 0):
                route = slim.conv2d(inputs, kernel_size=1, num_outputs=channels,
                                    scope=scope)
            else:
                route = slim.conv2d(inputs, kernel_size=3, num_outputs=channels,
                                    rate=rate, scope=scope)
            routes.append(route)
        net = tf.concat(routes, axis=3, name='concat')
        outputs = tf.identity(net, name='identity_output')
    return outputs


def upconcatblock(inputs, layer, up_layer, num_preblocks=2,
                  is_compress=False, is_upsample=True,
                  is_concat=True, is_concat_compress=False, scope=None):
    default_scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope, default_scope, [inputs, layer, up_layer]) as scope:
        net = tf.identity(inputs, name='identity_inputs')
        layer = tf.identity(layer, name='identity_layer')
        up_layer = tf.identity(up_layer, name='identity_uplayer')
        shape = layer.get_shape().as_list()
        up_shape = up_layer.get_shape().as_list()
        if num_preblocks > 0:
            net = slim.conv2d(net, kernel_size=1, num_outputs=up_shape[3], scope='conv')
            net = slim.conv2d(net, kernel_size=3, num_outputs=shape[3], scope='conv_1')
            for i in range(num_preblocks - 1):
                scope = 'block' if i == 0 else 'block_%d' % (i)
                net = bottleneckblock(net, scope=scope)
        if is_compress:
            net = slim.conv2d(net, kernel_size=3, num_outputs=up_shape[3], scope='conv_2')
        if is_upsample:
            net = upsample_block(net, up_shape)
        if is_concat:
            if is_concat_compress:
                up_layer = slim.conv2d(up_layer, kernel_size=1, num_outputs=int(up_shape[3] / 2), scope='conv_3')
            net = tf.concat([net, up_layer], axis=3)
        outputs = tf.identity(net, name='identity_output')
    return outputs


@slim.add_arg_scope
def scnnblock(inputs,
              activation_fn=tf.nn.relu,
              scope=None):
    default_scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
        inputs = tf.identity(inputs, name='identity_inputs')

        shape = inputs.get_shape().as_list()
        h_dim = 1
        w_dim = 2
        c_dim = 3
        channels = shape[c_dim]
        dims = [h_dim, w_dim]
        kernel_sizes = [[1, 3], [3, 1]]
        for k in range(2):  # h and w direction
            splits = tf.split(inputs, num_or_size_splits=shape[dims[k]], axis=dims[k])
            for j in range(2):  # forward and backward for one direction
                splits_t = []
                for i in range(len(splits)):
                    scope = 'conv_{0}_{1}'.format(k, j)
                    if i == 0:
                        net = splits[i]
                    else:
                        net = net + splits[i]
                        # pre-activate
                        if activation_fn is not None:
                            net = activation_fn(net)

                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None):
                        if i == 0:
                            net = slim.conv2d(
                                net, kernel_size=kernel_sizes[k], num_outputs=channels,
                                scope=scope)
                        else:
                            net = slim.conv2d(
                                net, kernel_size=kernel_sizes[k], num_outputs=channels,
                                reuse=True,
                                scope=scope)

                        net_t = tf.identity(net)
                        if activation_fn is not None:
                            net_t = activation_fn(net_t)
                    splits_t.append(net_t)
                splits = splits_t
                splits.reverse()
            inputs = tf.concat(splits, axis=dims[k])

        outputs = tf.identity(inputs, name='identity_output')
    return outputs
