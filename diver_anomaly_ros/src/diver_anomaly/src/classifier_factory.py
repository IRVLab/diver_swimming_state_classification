#!/usr/bin/env python

from conv_lstm_typeA import CNNLSTM_LW
from conv_lstm_typeB import CNNLSTM_DN
from conv import CNN_CW
from conv_simple import CNN_SIMPLE
from decision_tree import TSF_C
from transformer import TRANSFORMER


class Classifier_Factory:
    """Factory pattern class that returns an object of type X

        Simplifies the ROS classifier node by permitting configuration
        in a YAML file that minimizes the need for creating multiple
        separate classifier style nodes.
    """    
    def __init__(self):
        pass

    def get_classifier_from_string(self, string_name):        
        if string_name == 'CnnLstmLw':
            return CNNLSTM_LW()
        elif string_name == 'CnnLstmDn':
            return CNNLSTM_DN()
        elif string_name == 'CnnCw':
            return CNN_CW()
        elif string_name == 'CnnSimple':
            return CNN_SIMPLE()
        elif string_name == 'Tsf':
            return TSF_C()
        elif string_name == 'Transformer':
            return TRANSFORMER()
