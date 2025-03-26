import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved generator
generator = load_model("srgan_generator_mse.h5", compile=False)
generator.compile(optimizer="adam", loss="mean_squared_error")

# Optional: check the model architecture
generator.summary()

# Load test LR image
lr_test = cv2.imread("dataset/test/lr/portrait/1.png")
lr_test = cv2.resize(lr_test, (112, 150))  # Ensure it's the correct size
lr_test = lr_test / 255.0
lr_test = np.expand_dims(lr_test, axis=0)  # Add batch dimension

# Predict SR image
sr_image = generator.predict(lr_test)[0]
sr_image = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)
# sr_image = (sr_image * 255).astype(np.uint8)

# Save output
cv2.imwrite("output_sr.png", sr_image)