#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:32:04 2019

@author: rick
"""

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops

from deepin.src.nets import blocks

slim = tf.contrib.slim





_SCOPE = globals().get('__name__').split('.')[-1]
_WEIGHT_DECAY = 0.00004
_NORMALIZER_PARAMS = {
    'decay': 0.9997000098228455,
    'epsilon': 0.0010000000474974513,
    'scale': True,
    'center': True
}
_NORMALIZER_FN = slim.batch_norm
_WEIGHTS_INITIALIZER = initializers.xavier_initializer()#tf.truncated_normal_initializer(stddev=0.1)
_BIASES_INITIALIZER = init_ops.zeros_initializer()
_ACTIVATION_FN = tf.nn.relu6


def net(inputs,
        weight_decay=_WEIGHT_DECAY,
        activation_fn=_ACTIVATION_FN,
        normalizer_fn= _NORMALIZER_FN,
        normalizer_params=_NORMALIZER_PARAMS,
        weights_initializer=_WEIGHTS_INITIALIZER,
        biases_initializer=_BIASES_INITIALIZER,
        reuse=tf.AUTO_REUSE,
        is_training=True,
        scope=None):
    print('Enter network building----------------------')
    
    normalizer_params['is_training'] = is_training
    
    endpoints = []
    default_scope = _SCOPE
    with tf.variable_scope(scope, default_scope, values=[inputs]) as scope:
        print('scope: {}'.format(scope.name))
        inputs = tf.identity(inputs)
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            weights_initializer=weights_initializer,
                            biases_initializer=biases_initializer,
                            reuse=reuse), \
             slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay)), \
             slim.arg_scope([slim.separable_conv2d], 
                            weights_regularizer=None), \
             slim.arg_scope([blocks.res101block, blocks.bottleneckblock],
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            biases_initializer=biases_initializer), \
             slim.arg_scope([slim.batch_norm], **normalizer_params):
                #stage 1---------------------------
                net = slim.conv2d(inputs, kernel_size=3, num_outputs=32, stride=2, scope='conv')
                endpoints.append(tf.identity(net, name='ep'))  #fan-out
            
                #stage 2---------------------------
                net = slim.conv2d(net, kernel_size=3, num_outputs=64, stride=2, scope='conv_1')
                net = slim.separable_conv2d(net, kernel_size=3, depth_multiplier=1,
                                            num_outputs=None,
                                            scope='dwconv')
                net = slim.conv2d(net, kernel_size=1, num_outputs=48, scope='conv_2')
                endpoints.append(tf.identity(net, name='ep_1'))  #fan-out
    
                #stage 3---------------------------
                net = slim.conv2d(net, kernel_size=3, num_outputs=96, stride=2, 
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=None,
                                  scope='conv_3')
                net = blocks.res101block(net, is_separable=True, scope='res')
                net = blocks.res101block(net, is_separable=True, scope='res_1')
                net = slim.batch_norm(net, scope='bn')
                net = activation_fn(net)
                net = blocks.bottleneckblock(net, num_outputs_neck=190,
                                             num_outputs=96,  
                                             is_separable=True, 
                                             skip_last_activation=True,
                                             skip_last_normalizer=True,
                                             skip_last_biases=True,
                                             scope='bneck')
                net = blocks.res101block(net, is_separable=True, scope='res_2')
                net = blocks.res101block(net, is_separable=True, scope='res_3')
                net = slim.batch_norm(net, scope='bn_1')
                net = activation_fn(net)
                endpoints.append(tf.identity(net, name='ep_2'))  #fan-out
    
                #stage 4---------------------------
                net = slim.conv2d(net, kernel_size=3, num_outputs=192, stride=2, 
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=None,
                                  scope='conv_4')
                net = blocks.res101block(net, is_separable=True, scope='res_4')
                net = blocks.res101block(net, is_separable=True, scope='res_5')
                net = blocks.res101block(net, is_separable=True, scope='res_6')
                net = blocks.res101block(net, is_separable=True, scope='res_7')
                net = slim.batch_norm(net, scope='bn_2')
                net = activation_fn(net)
                net = blocks.bottleneckblock(net, num_outputs_neck=374, 
                                             num_outputs=192, 
                                             is_separable=True, 
                                             scope='bneck_1')
                net = blocks.denseblock(net, level=4, growth_rate=16, is_separable=True, scope='dens')
                net = blocks.bottleneckblock(net, num_outputs_neck=384,
                                             num_outputs=256,  
                                             is_separable=True, 
                                             skip_last_activation=True,
                                             skip_last_normalizer=True,
                                             skip_last_biases=True,
                                             scope='bneck_2')
                net = blocks.res101block(net, is_separable=True, scope='res_8')
                net = blocks.res101block(net, is_separable=True, scope='res_9')
                net = slim.batch_norm(net, scope='bn_3')
                net = activation_fn(net)
                endpoints.append(tf.identity(net, name='ep_3'))  #fan-out
            
                #stage 5---------------------------
                net = slim.conv2d(net, kernel_size=3, num_outputs=512, stride=2, 
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=None,
                                  scope='conv_5')
                net = blocks.res101block(net, is_separable=True, scope='res_10')
                net = blocks.res101block(net, is_separable=True, scope='res_11')
                net = slim.batch_norm(net, scope='bn_4')
                net = activation_fn(net)
                net = blocks.bottleneckblock(net, num_outputs_neck=628, 
                                             num_outputs=512, 
                                             is_separable=True, 
                                             scope='bneck_3')
                net = blocks.denseblock(net, level=8, growth_rate=16, is_separable=True, scope='dens_1')
                net = blocks.bottleneckblock(net, num_outputs_neck=762,
                                             num_outputs=640,  
                                             is_separable=True, 
                                             skip_last_activation=True,
                                             skip_last_normalizer=True,
                                             skip_last_biases=True,
                                             scope='bneck_4')
                net = blocks.res101block(net, is_separable=True, scope='res_12')
                net = blocks.res101block(net, is_separable=True, scope='res_13')
                net = blocks.res101block(net, is_separable=True, scope='res_14')
                net = blocks.res101block(net, is_separable=True, scope='res_15')
                net = slim.batch_norm(net, scope='bn_5')
                net = activation_fn(net)
                net = blocks.bottleneckblock(net, num_outputs_neck=960, 
                                             num_outputs=512, 
                                             is_separable=True, 
                                             scope='bneck_5')
                endpoints.append(tf.identity(net, name='ep_4')) #fan-out
            
    return endpoints
            

                 
                 
                 