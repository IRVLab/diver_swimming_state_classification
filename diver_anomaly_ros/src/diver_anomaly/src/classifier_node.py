#!/usr/bin/env python3

from classifier_factory import Classifier_Factory
from utils import get_numpy_from_2d_multiarray
import rospy
import rospkg
from std_msgs.msg import Float64MultiArray, Int64, String
from sensor_msgs.msg import Image
from uoled_msgs.msg import StatusIndicator
from estimator_msgs.msg import InferenceListMsg
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import message_filters
import yaml
from pathlib import Path, PurePath
import numpy as np
import cv2 
from not_cv_bridge import cv2_to_imgmsg, imgmsg_to_cv2
import torch
# import torchvision

# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# from binary_cnn import Binary_Classifier
# import cv2
# from plotting_utils import draw_estimation_on_img, draw_time_on_img, draw_prediction_on_img, label_as_text

GREEN = [0., 1., 0.]
RED = [1., 0., 0.]
YELLOW = [1., 1., 0.]
WHITE = [1., 1., 1.]

# Time in seconds, not ROS time actual seconds
GROUND_TRUTH_TIME=57

class DiverAnomalyClassificationNode():
    def __init__(self):

        rospy.loginfo('Initializing DiverAnomalyClassificationNode')
        rospy.init_node('classification_node', anonymous=True)

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('diver_anomaly')

        self.is_torch_model = True # If the model requires PyTorch
        self.is_tx2 = True

        self.plot_ground_truth = True

        rospy.loginfo(f"[INFO] Found the package path {pkg_path}")

        if not self.is_tx2:
            from cv_bridge import CvBridge, CvBridgeError
            self.bridge = CvBridge()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_start = rospy.Time.now()

        config_file = PurePath(pkg_path,'config','diver_anomaly_config.yaml')
        rospy.loginfo(f"[INFO] Loading configuration data from {config_file}")

        configs = None

        with open(config_file.as_posix(), 'r') as file:
            configs = yaml.safe_load(file)

        classifier_name = configs['classifier_name']
        self.classifier = Classifier_Factory().get_classifier_from_string(classifier_name)
        self.classifier.config_file_path = pkg_path # Give it as a string 
        self.classifier.configure()

        if self.is_torch_model:
            self.classifier.model = self.classifier.model.to(self.device)
            self.classifier.model.eval()

        print(f"Created a Classifier of type {type(self.classifier).__name__}")
        if type(self.classifier).__name__ == 'NoneType':
            print("Incorrect configuration.", classifier_name,  "is not correct. Check config. file.")
        self.img = None
        self.global_img = None
        self.observation_count = 0
        self.N_observations = configs['n_observations_required']
        self.classification_history = []
        self.classification_history_global = []
        self.filter_strength = configs['filter_strength']

        # rospy.Subscriber(configs['sequence_topic'], Float64MultiArray, self.sequence_cb)
        rospy.Subscriber(configs['image_topic'], Image, self.img_cb, queue_size=1)
        rospy.Subscriber('/combined_inference', InferenceListMsg, self.sequence_cb, queue_size=10)

        self.anomaly_detected = False
        self.anomaly_observed_pub = rospy.Publisher('/diver_anomaly/anomaly_observed', String, queue_size=10)
        self.classification_pub = rospy.Publisher('/diver_anomaly/classification', String, queue_size=10)
        self.img_pub = rospy.Publisher('/diver_anomaly/anomaly_img', Image, queue_size=10)
        self.uoled_pub = rospy.Publisher('/uoled/footer_left', String, queue_size=10)
        self.uoled_pub_status = rospy.Publisher('uoled/status_3', StatusIndicator, queue_size=10)
        self.uoled_pub_txt = rospy.Publisher('/uoled/display_text', String, queue_size=10)

        rospy.loginfo("Completed setup of DiverAnomalyClassificationNode")

    def create_status_msg(self, diver_state):
        si = StatusIndicator()
        si.text_color = WHITE
        if diver_state == 'swim':
            si.text = 'OK'
            si.bg_color = GREEN
        elif diver_state == 'not swim.':
            si.text = 'NS'
            si.bg_color = RED
        return si

    def img_cb(self, msg):
        try:
            if self.is_tx2:
                self.img = imgmsg_to_cv2(msg)
            else:
                self.img = np.array(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
            # self.global_img = np.copy(self.img)            
        except CvBridgeError:
            print("Failed on CvBridge") 

    def sequence_cb(self, msg):

        # img = np.copy(self.img)
        # print(f"Message received is {msg}")
        data_sequence = get_numpy_from_2d_multiarray(msg.feature_vec)
        # print(f"Message after converting to numpy is {data_sequence}")

        if self.is_torch_model:
            pred = self.classifier.classify(torch.from_numpy(data_sequence).to(self.device).unsqueeze(0))
            result = torch.argmax(pred).cpu().numpy()
        else:
            unsqueezed = torch.from_numpy(data_sequence).to(self.device).unsqueeze(0) # batch x time x features
            unsqueezed = unsqueezed.cpu().detach().numpy()
            pred = self.classifier.classify(unsqueezed)
            print(f"Prediction shape of TSF is {pred.shape}")

        pred = int(result)
        self.classification_history.append(pred)
        self.classification_history_global.append(pred)
        self.classification_pub.publish(str(pred))
        if self.observation_count < self.N_observations:
            rospy.loginfo(f"Accumulating observation {self.observation_count+1}/{self.N_observations}...")
            # $DEBUG Add back in once classifiers are actually working...
            # pred = np.int64(np.round(np.random.rand()))
        else:
            # self.observation_count = 0
            # For sliding window classification, we pop the first element, because we've added new
            # print(f"**********Before popping shape is {len(self.classification_history)}")
            self.classification_history.pop(0)
            self.anomaly_observed()

        # rospy.loginfo(f"Received a message with {len(msg.image_list)} total images ")
        for msg_id in msg.image_list:
            if self.is_tx2:
                img = imgmsg_to_cv2(msg_id)
            else:
                img = self.bridge.imgmsg_to_cv2(msg_id, desired_encoding="bgr8")
            img = self.decorate_img_with_time(img, msg.header.stamp)
            # if self.plot_ground_truth:
            #     if np.absolute(msg.header.stamp.to_sec() - GROUND_TRUTH_TIME) < 3: # epsilon
            #         img = self.decorate_image_with_ground_truth(img)

            img = self.decorator_wrapper(img)
            if self.is_tx2:
                self.img_pub.publish(cv2_to_imgmsg(img))
            else:
                self.img_pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
        
        self.observation_count += 1

    def anomaly_observed(self):
        self.anomaly_detected = False
        tmp = np.array(self.classification_history)
        sz = len(tmp)
        rospy.loginfo(f"Sliding window size {tmp.shape}")
        for i in range(self.filter_strength, sz-self.filter_strength):
            nn_prior = tmp[i-self.filter_strength]
            nn_prior_pred = self.get_avg_nn_pred(nn_prior)
            nn_future = tmp[i+self.filter_strength]
            nn_future_pred = self.get_avg_nn_pred(nn_future)
            if nn_prior_pred == 1 and nn_future_pred == 0: # Transition from swimming to not swimming
                rospy.loginfo("[DAD] Anomaly found in classification history!")
                self.anomaly_detected = True
                break

        msg = String()
        msg.data = str(self.anomaly_detected)
        self.anomaly_observed_pub.publish(msg)

        # anomaly_detected = True
        diver_state = "not swimming"
        if self.anomaly_detected:
            si = self.create_status_msg(diver_state)
            rospy.loginfo(f"[DAD] ======== Not swimming detected")
        else:
            diver_state = "swimming"
            si = self.create_status_msg(diver_state)
            rospy.loginfo(f"[DAD] ======== Swimming detected")

        self.uoled_pub_status.publish(si)
        self.uoled_pub_txt.publish(diver_state)
        # self.classification_history = []

    def decorate_image_with_legend(self, img):
        H, W, C = img.shape
        SQ_WIDTH = int(np.round(0.05*W)) # How wide to make legend squares
        SPACING = int(np.round(0.02*W)) # How much space between square and text x-direction
        TOP_Y = int(np.round(0.90*H))
        TOP_X = 10
        TEXT_SHIFT = int(np.round(0.45*W))
        BOTTOM_X = TOP_X+SQ_WIDTH
        BOTTOM_Y = TOP_Y+SQ_WIDTH
        if img is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale
            fontScale = 1
            # Blue color in BGR  RGB Cyan-Orange 192 57 43
            color = (0, 255, 255)
            # Line thickness of 2 px
            thickness = 2

            top = (TOP_X, TOP_Y)
            bottom = (BOTTOM_X, BOTTOM_Y)
            img = cv2.rectangle(img, top, bottom, (0, 0, 255), thickness=-1)
            # Using cv2.putText() method
            BOTTOM_TXT_X = BOTTOM_X+SPACING
            BOTTOM_TXT_Y = BOTTOM_Y
            img = cv2.putText(img, "Not swimming", (BOTTOM_TXT_X, BOTTOM_Y), font, 
                            fontScale, (0,0,255), thickness, cv2.LINE_AA)       
            TOPX2 = BOTTOM_TXT_X + TEXT_SHIFT  
            BOTTOMX2 = TOPX2+SQ_WIDTH     
            top = (TOPX2, TOP_Y)
            bottom = (BOTTOMX2, BOTTOM_Y)
            img = cv2.rectangle(img, top, bottom, (0, 255, 0), thickness=-1)
            BOTTOM_TXT_X2 = BOTTOMX2 + SPACING
            img = cv2.putText(img, "Swimming", (BOTTOM_TXT_X2, BOTTOM_Y), font, 
                            fontScale, (0,255,0), thickness, cv2.LINE_AA) 
        return img  

    def decorate_image_with_observation_count(self, img):
        H, W, C = img.shape
        TOP_Y = int(np.round(0.18*H))
        if img is not None:
            txt = 'Accumulating observations...' + str(self.observation_count) + "/" + str(self.N_observations)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (10, TOP_Y)

            # fontScale
            fontScale = 1

            # Blue color in BGR  RGB Cyan-Orange 192 57 43
            color = (0, 255, 255)

            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            img = cv2.putText(img, txt, org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
        return img

    def decorate_img_with_anomaly_status(self, img):
        H, W, C = img.shape
        TOP_Y = int(np.round(0.40*H))
        if img is not None:
            txt = ""
            if self.observation_count < self.N_observations:
                txt = "Anomaly detected: ... "

            else:
                txt = "Anomaly detected: " + str(self.anomaly_detected)
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (10, TOP_Y)
            # fontScale
            fontScale = 1
            color = (0, 255, 255)
            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            img = cv2.putText(img, txt, org, font, fontScale, color, thickness, cv2.LINE_AA)  
        return img         

    def decorate_img_with_sequence_bounding_box(self, img):
        buffer = 10 # pixels between bounding box and state diagram
        H, W, C = img.shape
        CELL_WIDTH = int(np.round(0.01*W))
        CELL_HEIGHT = int(np.round(0.05*H))
        TOP_Y = int(np.round(0.25*H))
        if img is not None:
            width = self.N_observations*CELL_WIDTH
            if self.observation_count > self.N_observations:
                tot_acc = len(self.classification_history_global)-1
                # tot_acc = len(self.classification_history_global)-len(self.classification_history)-1
                bottom = (((tot_acc-self.N_observations)*CELL_WIDTH)+10,TOP_Y+CELL_HEIGHT+buffer)
                top = ((tot_acc*CELL_WIDTH)+10, TOP_Y-buffer)
            else:
                top = (10,TOP_Y-buffer)
                bottom = ((self.N_observations*CELL_WIDTH)+10, TOP_Y+CELL_HEIGHT+buffer)

            img = cv2.rectangle(img, top, bottom, (0,255,255), thickness=3)
        return img

    def decorate_img_with_state(self, img):
        H, W, C = img.shape
        x_shift = int(np.round(0.01*W))
        y_shift = int(np.round(0.05*H))
        top_x = 10
        top_y = int(np.round(0.25*H))
        bottom_x = top_x + x_shift
        bottom_y = top_y + y_shift
        if img is not None:
            for i in self.classification_history_global:
                if i == 1:
                    # BGR Swimming = Green
                    color = (0, 255, 0)
                else:
                    # BGR Not Swimming = Red
                    color = (0, 0, 255)
                img = cv2.rectangle(img, (top_x, top_y), (bottom_x, bottom_y), color, thickness=-1)
                top_x += x_shift
                bottom_x += x_shift

        return img

    def decorate_img_with_classifier(self, img):
        H, W, C = img.shape
        X_ORG = int(np.round(0.65*W))
        Y_ORG = int(np.round(0.10*H))
        if img is not None:
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (X_ORG, Y_ORG)

            # fontScale
            fontScale = 1
            # Blue color in BGR  RGB Cyan-Orange R192 G57 B43
            color = (0, 255, 255)

            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            img = cv2.putText(img, str(type(self.classifier).__name__), org, font, 
                            fontScale, color, thickness, cv2.LINE_AA) 

        return img

    def decorate_img_with_time(self, img, t):
        H, W, C = img.shape
        Y_ORG = int(np.round(0.10*H))
        if img is not None:
            # t_now = rospy.Time.now()
            txt = "Elapsed time: " + str(np.round((t-self.time_start).to_sec(),2)) + " s"
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (10, Y_ORG)
            # fontScale
            fontScale = 1
            # Blue color in BGR  RGB Cyan-Orange R192 G57 B43
            color = (0, 255, 255)
            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            img = cv2.putText(img, txt, org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)          

        return img

    def decorator_wrapper(self, img):
        img = self.decorate_image_with_legend(img)
        img = self.decorate_img_with_classifier(img)
        img = self.decorate_img_with_sequence_bounding_box(img)
        img = self.decorate_image_with_observation_count(img)
        img = self.decorate_img_with_state(img)
        img = self.decorate_img_with_anomaly_status(img)
        return img

    def get_avg_nn_pred(self, input_nn):
        """ [1 1 1 0 1] -> (4/5) 0.80 = 1 """
        return int(np.round(np.mean(input_nn)))

if __name__ == '__main__':

    classifier_node = DiverAnomalyClassificationNode()
    classifier_node.uoled_pub.publish("")
    classifier_node.uoled_pub_txt.publish("")
    while not rospy.is_shutdown():
        classifier_node.uoled_pub.publish("DiverAnom.")
        rospy.spin()

