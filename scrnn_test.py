import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from scrnn import ExpandDimsLayer

 

# ✅ Load the Model with Custom Layer Registered
custom_objects = {"ExpandDimsLayer": ExpandDimsLayer}
scrnn = load_model("scrnn_model.h5", custom_objects=custom_objects, compile=False)
scrnn.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
scrnn._name = "SCRNN_Model"
scrnn.summary()

# # Load the saved SCRNN model
# scrnn = load_model("result/scrnn_model.h5")

# Load the model without compiling
# custom_objects = {"tf": tf}
# #scrnn = load_model("result/scrnn_model.h5", compile=False, custom_objects=custom_objects)
# scrnn = load_model("scrnn_model.h5", compile=False, custom_objects=custom_objects)

# # Recompile manually with the correct loss function
# scrnn.compile(optimizer="adam", loss="mean_squared_error")

# # Print summary to verify the model is loaded
# scrnn.summary()


# Load a new low-resolution image
lr_test = cv2.imread("dataset/test/lr/portrait/1.png") / 255.0  # Normalize
lr_test = np.expand_dims(lr_test, axis=0)  # Add batch dimension

# Predict high-resolution image
hr_predicted = scrnn.predict(lr_test)

# Convert back to an image format
hr_predicted = (hr_predicted[0] * 255).astype(np.uint8)

# Save or display the output
cv2.imwrite("output_hr.png", hr_predicted)
# cv2.imshow("Super-Resolved Image", hr_predicted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape               ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 150, 112, 3)        │               0 │
├──────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ conv2d (Conv2D)                      │ (None, 150, 112, 64)       │           4,864 │
├──────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ expand_dims_layer (ExpandDimsLayer)  │ (None, 1, 150, 112, 64)    │               0 │
├──────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ conv_lstm2d (ConvLSTM2D)             │ (None, 1, 150, 112, 64)    │         295,168 │
├──────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ conv_lstm2d_1 (ConvLSTM2D)           │ (None, 150, 112, 64)       │         295,168 │
├──────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ up_sampling2d (UpSampling2D)         │ (None, 300, 224, 64)       │               0 │
├──────────────────────────────────────┼────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 300, 224, 3)        │           4,803 │
└──────────────────────────────────────┴────────────────────────────┴─────────────────┘
 Total params: 600,003 (2.29 MB)
 Trainable params: 600,003 (2.29 MB)
 Non-trainable params: 0 (0.00 B)"""