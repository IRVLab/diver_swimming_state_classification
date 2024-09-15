#!/bin/bash 
YOLOv8_MODEL_WEIGHT="./weights/yolov8_v3_opset11.onnx"
VIDEOPOSE_MODEL_WEIGHT="./weights/videopose_opset11.onnx"
PATH_TO_CALIB_FILE="./data/calibs/zed_2024-06-16-19-15-33_4x6-camchain.yaml"
PATH_TO_IMAGES_DATA="/mnt/data/2024_08_19_Training_Anomaly_Detection/train_data"
OUTPUT_FOLDER="/mnt/data/2024_08_19_Training_Anomaly_Detection/output"

python gen_data.py \
    --estimator-model-path $YOLOv8_MODEL_WEIGHT \
    --liftor-model-path $VIDEOPOSE_MODEL_WEIGHT \
    --calib-path $PATH_TO_CALIB_FILE \
    --data-path $PATH_TO_IMAGES_DATA \
    --dst-path $OUTPUT_FOLDER

