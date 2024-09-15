#!/usr/bin/env python3

import torch
print(f"Torch CUDA version {torch.version.cuda}")
if torch.cuda.is_available():
	print("CUDA is available")
else:
	print("CUDA is not available")

import cv_bridge
print("Successfully import cv_bridge")

from cv_bridge.boost.cv_bridge_boost import getCvType
print("Successfully imported cv_bridge.boost.cv_bridge_boost")

import cv2
print(f"cv2 imported. Version {cv2.__version__}")

import onnxruntime
print(f"onnxruntime imported. Version {onnxruntime.__version__}")
print(f"onnruntime device {onnxruntime.get_device()}")

import sklearn
print(f"sklearn version {sklearn.__version__}")
