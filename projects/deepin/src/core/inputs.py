#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:34:48 2019

@author: rick
"""
from abc import abstractmethod



class Inputs(object):
    """Abstract base class for models"""  
    @abstractmethod
    def _build_data(self, *args, **kwargs):
        """Build data"""
        pass
      
    @abstractmethod
    def get(self, *args, **kwargs):
        """Get input tensors queue"""
        pass

    
    
    
    
    