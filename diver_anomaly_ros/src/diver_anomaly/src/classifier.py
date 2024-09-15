#!/usr/bin/env python3

# python2
# from abc import ABCMeta, abstractmethod
from abc import ABC, abstractmethod

class Classifier(ABC):
    """Abstract class that defines a classification method

        The idea is simply a definition of an interface that
        allows a single prediction method to class the classifier
        method.
    """
    #python2
    #__metaclass__ = ABCMeta

    @abstractmethod
    def classify(self, feature_sequence):
        """ Abstract definition for classifying a sequence"""
        pass
    @abstractmethod
    def configure(self):
        """ Abstract definition for configuring the classifier with config file"""
        pass


