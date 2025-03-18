import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, ConvLSTM2D, UpSampling2D, Lambda
from keras.models import Model
import os
import cv2
import numpy as np

"""
SCRNN arch
feature extraction
non-linear mapping
reconstruction

"""

# Define SCRNN with a fixed spatial input size
def build_scrnn(input_shape=(150, 112, 3)):  # Set fixed spatial dimensions (Height x Width)
    inputs = Input(shape=input_shape)

    # Feature extraction
    x = Conv2D(64, (5, 5), activation="relu", padding="same")(inputs)

    # ✅ Fix: Add an explicit time dimension
    # x = Lambda(lambda t: tf.expand_dims(t, axis=1))(x)  # Shape becomes (batch, 1, height, width, channels)
    x = Lambda(lambda t: tf.expand_dims(t, axis=1), output_shape=(1, input_shape[0], input_shape[1], 64))(x)

    # Non-linear mapping
    # ConvLSTM2D expects a 5D input: (batch, time, height, width, channels)
    x = ConvLSTM2D(64, (3, 3), activation="relu", padding="same", return_sequences=True)(x)
    x = ConvLSTM2D(64, (3, 3), activation="relu", padding="same", return_sequences=False)(x)

    # Reconstruction
    # Upscaling
    x = UpSampling2D(size=2)(x)  # Doubles the image size
    x = Conv2D(3, (5, 5), activation="sigmoid", padding="same")(x)  # Output HR image

    model = Model(inputs, x)
    return model

# Build and compile the model with the fixed input size
# 150 is height, 112 is width, and 3 is the number of channels (RGB)
scrnn = build_scrnn(input_shape=(150, 112, 3))  # Or (112, 150, 3) for portrait mode
scrnn.compile(optimizer="adam", loss="mse")
scrnn.summary()
"""
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                        ┃ Output Shape               ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)            │ (None, 150, 112, 3)        │               0 │
├─────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ conv2d (Conv2D)                     │ (None, 150, 112, 64)       │           4,864 │
├─────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ lambda (Lambda)                     │ (None, 1, 150, 112, 64)    │               0 │
├─────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ conv_lstm2d (ConvLSTM2D)            │ (None, 1, 150, 112, 64)    │         295,168 │
├─────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ conv_lstm2d_1 (ConvLSTM2D)          │ (None, 150, 112, 64)       │         295,168 │
├─────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ up_sampling2d (UpSampling2D)        │ (None, 300, 224, 64)       │               0 │
├─────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                   │ (None, 300, 224, 3)        │           4,803 │
└─────────────────────────────────────┴────────────────────────────┴─────────────────┘
 Total params: 600,003 (2.29 MB)
 Trainable params: 600,003 (2.29 MB)
 Non-trainable params: 0 (0.00 B)
"""

