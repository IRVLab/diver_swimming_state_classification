#!/bin/bash python3

from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout, String
import numpy as np
import yaml
import rospy
from uoled_msgs.msg import StatusIndicator

BLACK = [0., 0., 0.]
def clear_uoled_display():
    status_pubs = [
        rospy.Publisher('/uoled/status_1', StatusIndicator, queue_size=1),
        rospy.Publisher('/uoled/status_2', StatusIndicator, queue_size=1),
        rospy.Publisher('/uoled/status_3', StatusIndicator, queue_size=1),
        rospy.Publisher('/uoled/status_4', StatusIndicator, queue_size=1),
        rospy.Publisher('/uoled/status_5', StatusIndicator, queue_size=1)
    ]
    si = StatusIndicator()
    si.text = ""
    si.text_color = BLACK
    si.bg_color = BLACK
    for i in range(4):
        status_pubs[i].publish(si)
    
    footer = rospy.Publisher('/uoled/footer_left', String, queue_size=1)
    footer.publish("")
# rospy.init_node('tmp_pubisher')
# clear_uoled_display()    

def load_config_parameters(file_path):
    configs = None
    with open(file_path.as_posix(), 'r') as file:
        configs = yaml.safe_load(file)
    return configs
    

def numpy_2d_to_msg(data_in):

    rows, cols = data_in.shape
    multi_array_dimension_list = []

    tmp = MultiArrayDimension()
    tmp.label = 'rows'
    tmp.size = int(rows)
    multi_array_dimension_list.append(tmp)

    tmp = MultiArrayDimension()
    tmp.label = 'cols'
    tmp.size = int(cols)
    multi_array_dimension_list.append(tmp)

    multi_array_layout = MultiArrayLayout()
    multi_array_layout.dim = multi_array_dimension_list

    data_array_msg = Float64MultiArray()
    data_array_msg.data = data_in.flatten()
    data_array_msg.layout = multi_array_layout

    return data_array_msg

def get_numpy_from_2d_multiarray(data_msg_in):
    data = np.array(data_msg_in.data).astype(np.float32)
    layout = data_msg_in.layout

    n_rows = None
    n_cols = None

    for array_dim in layout.dim:
        if array_dim.label == 'rows':
            n_rows = array_dim.size
        elif array_dim.label == 'cols':
            n_cols = array_dim.size

    return data.reshape((n_rows, n_cols))

