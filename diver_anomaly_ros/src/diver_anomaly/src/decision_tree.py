#!/usr/bin/env python3
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import rospy
from pathlib import Path, PurePath
from utils import load_config_parameters
import yaml
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from zipfile import ZipFile, ZipInfo, BadZipFile

from classifier import Classifier
from models.TSF import TSF
from aeon.datasets import load_from_tsfile
from aeon.classification.interval_based import TimeSeriesForestClassifier

def fixBadZipfile(zipFile):  
    print(f"***********Fixing zip file")
    f = open(zipFile, 'r+b')  
    data = f.read()  
    pos = data.rfind(b'\x50\x4b\x05\x06') # End of central directory signature  
    if (pos > 0):
        # self._log("Trancating file at location " + str(pos + 22)+ ".")  
        f.seek(pos + 22)   # size of 'ZIP end of central directory record' 
        f.truncate()  
        # f.write(b'\x00\x00')
        # f.seek(0) 
        f.close()

class TSF_C(Classifier):
    def __init__(self):
        self.config_file_path = None
        self.config_file = "decision_tree_config.yaml"
        self.model = None

    def configure(self):
        if self.config_file_path is not None:
            path_to_config_file = PurePath(self.config_file_path, 'config', self.config_file) 
            rospy.loginfo(f"Path to config file is {path_to_config_file.as_posix()}")
            configs = load_config_parameters(path_to_config_file)
        # self.model = TSF(num_trees=int(configs['n_trees']))
        self.model = TimeSeriesForestClassifier(n_estimators=int(configs['n_trees']),time_limit_in_minutes = 1)
        path_as_str = PurePath(self.config_file_path,'weights', configs['weight_file']).as_posix()
        rospy.loginfo(f"path to string is {path_as_str}")
        # fixBadZipfile('/home/dtkutzke/ros_workspaces/diver_anomaly_ros/src/diver_anomaly/weights/tsfmodelweightsclassification.zip')
        # z = ZipFile('/home/dtkutzke/ros_workspaces/diver_anomaly_ros/src/diver_anomaly/weights/tsfmodelweightsclassification.zip')
        try:
            self.model.load_from_path(path_as_str)
        except BadZipFile:
            print(f"File seems to be corrupted") 

        # rospy.loginfo(f"The model fitted? {self.model.check_is_fitted()}")
        rospy.loginfo(f"Path to the weight file is {PurePath(self.config_file_path,'weights', configs['weight_file']).as_posix()}")
        rospy.loginfo(f"Successfully initialized TSF")

    def classify(self,feature_sequence):
        print(f"Feature sequence dimensions {feature_sequence.shape}")
        result = self.model.predict_proba(feature_sequence)
        print(f"Result of time series forest is {result.shape} and {result}")
        return result