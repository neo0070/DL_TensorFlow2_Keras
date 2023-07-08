# coding=utf-8
#
# @file laboratory/pnuskgh/chapter_1_001.py
# @version 0.0.1
# @license OBCon License 1.0
# @copyright pnuskgh, All right reserved.
# @author gye hyun james kim <pnuskgh@gmail.com>

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print("------------------------------------------------------------")
print(tf.config.list_physical_devices("GPU"))


# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # --- 0. 0번 GPU 사용, -1. GPU 사용하지 않음
# with tf.device('/device:GPU:0'):
