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



def upsampleblock(inputs, newsize, scope=None):
    '''Args:
        inputs is a tensor with [batch, height, width, channels],
        newsize is a tensor with [height, width]'''
    default_scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
        inputs = tf.identity(inputs, name='input')
        net = tf.image.resize_bilinear(inputs, newsize)
        net = tf.identity(net, name='upsampled')
    return net 

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
    
#def res50block(inputs, compress_rate=2, scope=None):
#    default_scope = sys._getframe().f_code.co_name
#    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
#        inputs = tf.identity(inputs, name='identity_input')
#        input_channels = inputs.get_shape().as_list()[3]
#        channels = int(input_channels / compress_rate)
#        with slim.arg_scope([slim.conv2d],
#                            stride=1, 
#                            padding='SAME'):
#            net = slim.conv2d(inputs, kernel_size=1, num_outputs=channels, scope='conv2d')
#            net = slim.conv2d(net, kernel_size=3, num_outputs=input_channels, scope='conv2d_1')
#        net = tf.add(net, inputs, name='add')
#        outputs = tf.identity(net, name='identity_output')
#    return outputs
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
        #pre-activate
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

#def res101block(inputs, compress_rate=2, is_separable=False, scope=None):
#    default_scope = sys._getframe().f_code.co_name
#    with tf.variable_scope(scope, default_scope, [inputs]) as scope:
#        inputs = tf.identity(inputs, name='identity_input')
#        input_channels = inputs.get_shape().as_list()[3]
#        channels = int(input_channels / compress_rate)
#        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
#                            stride=1,
#                            padding='SAME'):
#            net = slim.conv2d(inputs, kernel_size=1, num_outputs=channels, scope='conv2d')
#            if is_separable:
#                net = slim.separable_conv2d(
#                        net, 
#                        kernel_size=3, 
#                        num_outputs=input_channels,
#                        depth_multiplier=1,
#                        scope='separable_conv2d')
#            else:
#                net = slim.conv2d(net, kernel_size=3, num_outputs=channels, scope='conv2d_1')
#                net = slim.conv2d(net, kernel_size=1, num_outputs=input_channels, scope='conv2d_2')
#        net = tf.add(net, inputs, name='add')
#        outputs = tf.identity(net, name='identity_output')
#    return outputs
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
        #pre-activate
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
            scope = 'conv2d' if i == 0 else 'conv2d_%d'%(i)
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
            for i in range(num_preblocks-1):
                scope = 'block' if i == 0 else 'block_%d'%(i)
                net = bottleneckblock(net, scope=scope)
        if is_compress:
            net = slim.conv2d(net, kernel_size=3, num_outputs=up_shape[3], scope='conv_2')
        if is_upsample:
            net = upsampleblock(net, up_shape[1:3])
        if is_concat:
            if is_concat_compress:
                up_layer = slim.conv2d(up_layer, kernel_size=1, num_outputs=int(up_shape[3]/2), scope='conv_3')
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
        for k in range(2): #h and w direction
            splits = tf.split(inputs, num_or_size_splits=shape[dims[k]], axis=dims[k])
            for j in range(2): #forward and backward for one direction
                splits_t = []
                for i in range(len(splits)):
                    scope = 'conv_{0}_{1}'.format(k, j)
                    if i == 0:
                        net = splits[i]
                    else:
                        net = net + splits[i]
                        #pre-activate
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
    
        
    
    