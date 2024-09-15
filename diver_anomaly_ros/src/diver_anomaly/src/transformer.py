#!/usr/bin/env python3

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pathlib import Path, PurePath

from classifier import Classifier
from models.attention import attentionMTSC
from utils import load_config_parameters
import torch
import yaml

class TRANSFORMER(Classifier):
    def __init__(self):
        self.config_file_path = None
        self.config_file = "transformer_config.yaml"
        self.model = None

    def configure(self):
        if self.config_file_path is not None:
            path_to_config_file = PurePath(self.config_file_path, 'config', self.config_file) 
            print(f"Path to config file is {path_to_config_file.as_posix()}")
            configs = load_config_parameters(path_to_config_file)
        print(f"Loaded all configuration parameters {configs}")
        self.model = attentionMTSC(
                series_len=int(configs['series_len']),
                input_dim=int(configs['input_dim']),
                learnable_pos_enc=bool(configs['learnable_pose_enc']),
                d_model=int(configs['d_model']),
                heads=int(configs['heads']),
                classes=int(configs['classes']),
                dropout=float(configs['dropout']),
                dim_ff=int(configs['dim_ff']),
                num_layers=int(configs['num_layers']),
                task=str(configs['task']))
        self.model.load_state_dict(torch.load(PurePath(self.config_file_path,'weights', configs['weight_file']).as_posix()))

    def classify(self,feature_sequence):
        result = self.model(feature_sequence)
        # change
        return result
    
