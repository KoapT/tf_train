#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:50:35 2019

@author: rick
"""

from yolo.src.models import object_detection

_model_dict = {
    'Object_detection': object_detection.ObjectDetection(),
}


def get_model(model_name):
    if model_name not in _model_dict:
        raise ValueError('Model "{}" is not registered.'.format(model_name))

    return _model_dict[model_name]
