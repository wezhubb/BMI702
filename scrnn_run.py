# from data_preprocessing import load_images_from_folder
from scrnn import build_scrnn
import os
import cv2
import numpy as np


def load_images_from_folder(folder, target_size=None):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)  # Read image
        if img is None:
            print(f"Skipping corrupted file: {filename}")
            continue
        # height, width, channels = img.shape
        # print(f"Loaded {filename}: Shape = {height} x {width} x {channels}")  # Print dimensions

        if target_size:
            img = cv2.resize(img, target_size)  # Resize if needed
        img = img / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images)


# Load HR and LR images (both landscape and portrait)
hr_landscape = load_images_from_folder("dataset/train/hr/landscape")
hr_portrait = load_images_from_folder("dataset/train/hr/portrait")
lr_landscape = load_images_from_folder("dataset/train/lr/landscape")
lr_portrait = load_images_from_folder("dataset/train/lr/portrait")

# 150 is height, 112 is width, and 3 is the number of channels (RGB)
scrnn = build_scrnn(input_shape=(150, 112, 3))  # Or (112, 150, 3) for portrait mode
scrnn.compile(optimizer="adam", loss="mse")
scrnn.fit(lr_portrait, hr_portrait, epochs=50, batch_size=8, validation_split=0.1)

# Evaluate model on training data
train_loss = scrnn.evaluate(lr_portrait, hr_portrait)
print(f"Training Loss: {train_loss}")

scrnn._name = "SCRNN_Model"

scrnn.save("scrnn_model.h5")