#!/usr/bin/env python3

from pathlib import Path, PurePath

from classifier import Classifier
from models.cnn_dn import CNNLSTM_dn
from utils import load_config_parameters
import torch
import yaml

class CNNLSTM_DN(Classifier):
    def __init__(self):
        self.config_file_path = None
        self.config_file = "lstm_typeB_config.yaml"
        self.model = None

    def configure(self):
        if self.config_file_path is not None:
            path_to_config_file = PurePath(self.config_file_path, 'config', self.config_file) 
            print(f"Path to config file is {path_to_config_file.as_posix()}")
            configs = load_config_parameters(path_to_config_file)
        self.model = CNNLSTM_dn(int(configs['input_dim']), configs['time_steps'], int(configs['num_filters']), int(configs['hidden_size']), int(configs['classes']))  
        self.model.load_state_dict(torch.load(PurePath(self.config_file_path,'weights', configs['weight_file']).as_posix()))

    def classify(self,feature_sequence):
        result = self.model(feature_sequence)
        # change
        return result
