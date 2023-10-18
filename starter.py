import torch

import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras_core as keras

print(keras.ops.zeros((1,)))