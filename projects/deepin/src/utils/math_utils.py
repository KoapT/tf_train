#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:48:17 2019

@author: rick
"""


import tensorflow as tf




def max_normalize(tensor):
    '''
    Normalize by dividing own max value per batch
    Args:
        tensor: tensorflow tensor with shape [batch, height, width, channels] or
            [height, width, channels]
    '''
    if tensor.shape.ndims == 4:
        max_values = tf.reduce_max(tensor, axis=1)
        max_values = tf.reduce_max(max_values, axis=1)
        max_values = tf.reduce_max(max_values, axis=1)
        max_values = tf.expand_dims(max_values, axis=1)
        max_values = tf.expand_dims(max_values, axis=1)
        max_values = tf.expand_dims(max_values, axis=1)
    elif tensor.shape.ndims == 3:
        max_values = tf.reduce_max(tensor)
    else:
        raise ValueError(
                'Input tensor shape must be [batch, height, width, channels] or'
                '[height, width, channels].')
    
    tensor = tf.divide(tensor, max_values)
    return tensor

def normalize(inputs, max_value, scale, shift):
    return (inputs / max_value) * scale - shift

def denormalize(inputs, max_value, scale, shift):
    return (inputs + shift) * max_value / scale

