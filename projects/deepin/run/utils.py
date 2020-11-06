#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:55:45 2019

@author: rick
"""
import os
import re
from tensorflow.python.lib.io import file_io





def unifyCheckPointTxt(path):
    if file_io.file_exists(path):
        file_content = file_io.read_file_to_string(path)
        file_content_new = ''
        cnt = 0
        for line in file_content.split('\n'):
            cnt = cnt + 1
            items = line.split(':')
            if (len(items) != 2) or (cnt > 2):
                break
            items[1] = re.sub('"(.*?)"', r'\1', items[1])
            items[1] = os.path.basename(items[1]).split('-')[0]
            file_content_new = file_content_new + items[0] + ': "' + items[1] + '"\n'
        file_io.write_string_to_file(path, file_content_new)



