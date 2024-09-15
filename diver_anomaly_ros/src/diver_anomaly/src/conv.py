#!/usr/bin/env python3

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pathlib import Path, PurePath

from classifier import Classifier
from models.cnn_cw import CNN_cw
from utils import load_config_parameters
import torch
import yaml

class CNN_CW(Classifier):
    def __init__(self):
        self.config_file_path = None
        self.config_file = "conv_config.yaml"
        self.model = None

    def configure(self):
        if self.config_file_path is not None:
            path_to_config_file = PurePath(self.config_file_path, 'config', self.config_file) 
            print(f"Path to config file is {path_to_config_file.as_posix()}")
            configs = load_config_parameters(path_to_config_file)
        self.model = CNN_cw(int(configs['input_dim']), int(configs['time_steps']), int(configs['classes']), int(configs['num_filters1']), int(configs['num_filters2']), int(configs['pool1']),int(configs['pool2']),int(configs['dimff']))  
        print(f" Path to the weights: {PurePath(self.config_file_path,'weights', configs['weight_file']).as_posix()}")
        self.model.load_state_dict(torch.load(PurePath(self.config_file_path,'weights', configs['weight_file']).as_posix()))
    

    def classify(self,feature_sequence):
        result = self.model(feature_sequence)
        return result
        # $DEBUG Do whatever I need to for turning
