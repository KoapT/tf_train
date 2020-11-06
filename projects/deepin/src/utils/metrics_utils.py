#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:11:46 2019

@author: rick
"""

import tensorflow as tf




def accuracy_multi_label(groundtruth_logits, prediction_logits, threshold):
    groundtruth_logits = tf.greater(groundtruth_logits, threshold)
    prediction_logits = tf.greater(prediction_logits, threshold)
    correct_flags = tf.reduce_all(
            tf.equal(groundtruth_logits, prediction_logits), axis=1)
    groundtruth_logits = tf.ones_like(correct_flags, tf.int32)
    prediction_logits = tf.where(correct_flags, 
                                 groundtruth_logits, 
                                 tf.zeros_like(groundtruth_logits, tf.int32))
    accuracy = tf.metrics.accuracy(
            labels=groundtruth_logits, predictions=prediction_logits,
            weights=tf.ones(groundtruth_logits.shape, dtype=tf.float32))
    return accuracy