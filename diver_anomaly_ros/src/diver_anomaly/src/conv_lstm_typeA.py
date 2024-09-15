#!/usr/bin/env python3

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import rospy
from pathlib import Path, PurePath

from classifier import Classifier
from models.cnn_lw import CNNLSTM_lw
from utils import load_config_parameters
import torch
import yaml
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

class CNNLSTM_LW(Classifier):
    def __init__(self):
        self.config_file_path = None
        self.config_file = "lstm_typeA_config.yaml"
        self.model = None
        
    def configure(self):
        if self.config_file_path is not None:
            path_to_config_file = PurePath(self.config_file_path, 'config', self.config_file) 
            rospy.loginfo(f"Path to config file is {path_to_config_file.as_posix()}")
            configs = load_config_parameters(path_to_config_file)
        self.model = CNNLSTM_lw(int(configs['input_dim']), int(configs['num_filters']), int(configs['hidden_size']), int(configs['classes']))  
        rospy.loginfo(f"Path to the weight file is {PurePath(self.config_file_path,'weights', configs['weight_file']).as_posix()}")
        self.model.load_state_dict(torch.load(PurePath(self.config_file_path,'weights', configs['weight_file']).as_posix()))
        rospy.loginfo(f"Successfully initialized ConvLstmA")

    def classify(self,feature_sequence):
        result = self.model(feature_sequence)
        # change
        return result

#Classifier.register(ConvLstmA)
