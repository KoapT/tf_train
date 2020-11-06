#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:46:06 2019

@author: rick
"""

import tensorflow as tf
import six


def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

def norm2bytes_list(value):
    for i in range(len(value)):
        value[i] = value[i].encode() if isinstance(value[i], str) and six.PY3 else value[i]
    return value

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(value)]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=norm2bytes_list(value)))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

 
