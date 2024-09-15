#!/bin/bash python3

import cv2
print(f"Successfully imported OpenCV version {cv2.__version__}")

import torch
print(f"Successfully imported PyTorch version {torch.__version__}")
print(f"Torch says CUDA available = {torch.cuda.is_available()}")

import onnxruntime
print(f"Successfully imported onnxruntime version {onnxruntime.__version__}")
print(f"Onnxruntime says device available is {onnxruntime.get_device()}")

