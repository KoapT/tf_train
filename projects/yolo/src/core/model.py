#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:33:01 2019

@author: rick
"""


from abc import abstractmethod
import collections

TrainItem = collections.namedtuple(
        'TrainItem',
        [
         'loss',  # A loss tensor corresponding a optimizer.
         'var_list',  # A trainable variable list applied to gradients of the loss,
                      # if it is none, all the trainable variables in the graph will be as default,
                      # such as tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES).
         'summary_name', # A name for summary.
        ])

BuildModelOutputs = collections.namedtuple(
        'BuildModelOutputs',
        [
         'train_item_list',  # A list of TrainItem(), [TrainItem1, TrainItem2, ...].
         'metric_dict',  # A dictionary of metrics, {'...': some metric op, ...}.
        ])

OptimizerItem = collections.namedtuple(
        'OptimizerItem',
        [
         'optimizer',  # A optimizer corresponding a loss.
         'learn_rate',  # A learning rate of the optimizer.
         'summary_name', # A name for summary.
        ])

BuildOptimizerOutputs = collections.namedtuple(
        'BuildOptimizerOutputs',
        [
         'optimizer_item_list',  # A list of OptimizerItem(), [OptimizerItem, OptimizerItem, ...].
        ])


class Model(object):
    """Abstract base class for models"""
    
    def __init__(self):
        """Constructor"""
        pass
    
    @abstractmethod
    def initialize(self, model_info_path, sample_info_path, is_training):
        """Initialize"""
        pass
#Unnecessary================    
#    @abstractmethod
#    def preprocess(self, inputs):
#        """Inputs preprocessing"""
#        pass
#    
#    @abstractmethod
#    def predict(self, preprocessed_inputs):
#        """Predict predictions from inputs preprocessed"""
#        pass
#    
#    @abstractmethod
#    def postprocess(self, predictions):
#        """Convert predictions to final outputs"""
#        pass
#    
#    @abstractmethod
#    def loss(self, predictions, provided_groundtruths):
#        """Compute loss of predictons with respect to groundtruth provided"""
#        pass
#    
#    @abstractmethod
#    def provide_groundtruth(self, groundtruths):
#        """Provide groundtruths"""
#        pass
#=============================    
    @abstractmethod
    def build_inputs(self, sample_path, num_samples, batch_size, num_clones):
        """Build inputs, get input samples queue.
           Note: this method will be excuted before build_model().
        """
        pass
    
    @abstractmethod
    def build_model(self, input_queue):
        """Build model getting samples from input_queue, and can add summaries.
           It must return train_dict_list and optionally metric_dict.
           Note: this method will be excuted before build_optimizers().
        """
        
        """A list of TrainItem(), [TrainItem1, TrainItem2, ...].
           It represents multiple losses as list.
           First loss and ver_list of TrainItem must be built by build_model(),
            then optimizer corresponding loss of TrainItem must be built by build_optimizer().
           Note: every optimizer is corresponding to a learning_rate, a loss and a ver_list.
        """
        train_item_list = []
        
        """{'...': some metric op, ...}"""
        metric_dict = {}
        
        outputs = BuildModelOutputs(train_item_list=train_item_list, 
                                    metric_dict=metric_dict)
        
        return outputs
    
    @abstractmethod
    def build_optimizer(self, train_number_of_steps):
        """Build optimizers with corresponding learning rates, and can add summaries.
           It must return optimizer_list.
           Note: this method will be excuted after build_model().
        """
        
        """A list of OptimizerItem(), [OptimizerItem, OptimizerItem, ...].
           The optimizer must be corresponding to loss in train_item_list returned by build_model()"""
        optimizer_item_list = []
        
        outputs = BuildOptimizerOutputs(optimizer_item_list=optimizer_item_list)
        
        return outputs
    
    @abstractmethod
    def get_batch_size(self):
        """Get batch size which is load in model_info_path from initialize()"""
        pass
    
    @abstractmethod
    def create_for_inferrence(self):
        """Create model with io for inferrence, 
            specify input and output scope"""
        pass
    
    @abstractmethod
    def get_input_names(self):
        """Get input tenor names"""
        pass
    
    @abstractmethod
    def get_output_names(self):
        """Get output tensor names"""
        pass
    
    @abstractmethod
    def get_extra_layer_scopes(self):
        """Get the layer scopes whose weights want to be initialized randomly,
           don't need to be restored from checkpoint in training"""
        pass
    
    @abstractmethod
    def inference(self, graph, sess, feeds, model_info_path, sample_info_path):
        """Inferrence with frozen_inference_graph.pb"""
        pass
    
    @abstractmethod
    def schedule_per_train_step(self, train_op_list, step):
        """Get some train ops from train_op_list in every train step for run session,
            return train_op, it can be a loss tensor or a loss tensor list.
           Note: train_op_list is actually a loss tensor list,
            which is corresponding with train_item_list built by build_model().
        """
        train_op = train_op_list
        return train_op
    
    @abstractmethod
    def every_before_train_step_callback_fn(self, sess):
        """This function will be callback on every train step,
           before train op sess.run
           You can define the tensor name in build_model(), and
               place some debug code in this function, such as:
               tensor = sess.graph.get_tensor_by_name('tensor name:0') 
               for sess.run([tensor]), to test.
           You can also place some wait code in it, such as:
               cv2.waitKey(0).
           If you do not want to use it, place pass in it."""
        pass
    
    @abstractmethod
    def every_after_train_step_callback_fn(self, sess):
        """This function will be callback on every train step,
           after train op sess.run
           You can define the tensor name in build_model(), and
               place some debug code in this function, such as:
               tensor = sess.graph.get_tensor_by_name('tensor name:0') 
               for sess.run([tensor]), to test.
           You can also place some wait code in it, such as:
               cv2.waitKey(0).
           If you do not want to use it, place pass in it."""
        pass
    
    
