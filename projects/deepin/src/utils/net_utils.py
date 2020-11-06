#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:48:17 2019

@author: rick
"""
import tensorflow as tf





def weight(shape):
    return tf.Variable(
            initial_value=tf.truncated_normal(shape=shape, stddev=0.1), 
            dtype=tf.float32)

def bias(shape):
    return tf.Variable(
            initial_value=tf.constant(value=0.1, shape=shape), 
            dtype=tf.float32)

def conv(x, w, strides=1, padding='SAME'):
    return tf.nn.conv2d(
            x, w, strides=[1, strides, strides, 1], 
            padding=padding)

def conv_depthwise(x, w, strides=1, padding='SAME'):
    return tf.nn.depthwise_conv2d(
            x, w, strides=[1, strides, strides, 1], 
            padding=padding)

def conv_separable(x, w_d, w_p, strides=1, padding='SAME'):
    return tf.nn.separable_conv2d(
            x, w_d, w_p, strides=[1, strides, strides, 1],
            padding=padding)

def bn(x, is_training=True):
    return tf.layers.batch_normalization(x, training=is_training)
    
def activate(x):
    return tf.nn.relu(x)#tf.nn.sigmoid(x)

def depth_to_space(x, block_size):
    return tf.depth_to_space(x, block_size)

def upsample(x, newsize):
    x = tf.image.resize_bilinear(x, newsize)
    x = tf.identity(x, name='upsampled')
    return x   

def softmax(x):
    return tf.nn.softmax(x)