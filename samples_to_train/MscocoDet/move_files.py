#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:41:59 2019

@author: rick
"""

import os
import shutil
import tensorflow as tf



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('src_dir', '',
                     'source data loaded directory')
tf.app.flags.DEFINE_string('dest_dir_img', '',
                     'destination directory of image file')
tf.app.flags.DEFINE_string('dest_dir_anno', '',
                     'destination directory of annotation file')


fileNameExts = ['.jpg', '.jpeg']


def listAllFiles(dir):
    pathList = []
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isdir(path):
            pathList.extend(listAllFiles(path))
        if os.path.isfile(path):
            pathList.append(path)
    return pathList

if __name__ == '__main__':
    for file in listAllFiles(FLAGS.src_dir):
        fileDir = os.path.dirname(file)
        fileName = os.path.basename(file)
        fileNameBase = os.path.splitext(fileName)[0]
        fileNameExt = os.path.splitext(fileName)[1].lower()
        if (fileNameExt in fileNameExts):
            annotationFileName = fileNameBase+'.xml'
            annotationFile = os.path.join(fileDir, annotationFileName)
            if os.path.exists(annotationFile):
                #right pair to load
                shutil.move(file, os.path.join(FLAGS.dest_dir_img, fileName))
                shutil.move(annotationFile, os.path.join(FLAGS.dest_dir_anno, annotationFileName))
                



