import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil

# Define paths
data_folder = "data"  # Folder where HR images are stored
output_folder = "dataset"  # Folder to save train/test HR and LR images
downsample_factor = 2  # Set downsampling factor (e.g., 2x or 4x)

# Create required directories
for split in ["train", "test"]:
    for res in ["hr", "lr"]:
        os.makedirs(os.path.join(output_folder, split, res), exist_ok=True)

# Function to check if an image is valid
def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Check for corruption
        return True
    except Exception as e:
        print(f"Skipping invalid image: {image_path} - Error: {e}")
        return False

# Function to load and downsample images dynamically
def load_and_downsample_images(folder, downsample_factor=2):
    hr_images, lr_images, filenames = [], [], []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        
        # Check if file is valid
        if not is_valid_image(img_path):
            continue  # Skip invalid images
        
        try:
            # Load high-resolution (HR) image
            hr_img = load_img(img_path)  
            hr_img = np.array(hr_img) / 255.0  # Normalize
            
            # Get original size
            original_h, original_w, _ = hr_img.shape  
            
            # Compute new downsampled size
            lr_w, lr_h = original_w // downsample_factor, original_h // downsample_factor
            
            # Downsample image using OpenCV
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

            # Store HR and LR images
            hr_images.append(hr_img)
            lr_images.append(lr_img)
            filenames.append(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return hr_images, lr_images, filenames  # Keep lists, not NumPy arrays

# Load and downsample images dynamically
hr_images, lr_images, filenames = load_and_downsample_images(data_folder, downsample_factor=2)

# Split into train (80%) and test (20%)
train_hr, test_hr, train_lr, test_lr, train_filenames, test_filenames = train_test_split(
    hr_images, lr_images, filenames, test_size=0.2, random_state=42
)

# Function to save images
def save_images(images, filenames, folder):
    for img, filename in zip(images, filenames):
        save_path = os.path.join(folder, filename)
        cv2.imwrite(save_path, (img * 255).astype(np.uint8))  # Convert back to uint8

# Save HR and LR images in respective folders
save_images(train_hr, train_filenames, os.path.join(output_folder, "train", "hr"))
save_images(train_lr, train_filenames, os.path.join(output_folder, "train", "lr"))
save_images(test_hr, test_filenames, os.path.join(output_folder, "test", "hr"))
save_images(test_lr, test_filenames, os.path.join(output_folder, "test", "lr"))


print("Dataset has been successfully split and saved!")

# Function to classify and move images based on size
def classify_and_move_images():
    splits = ["train", "test"]  # Both training and testing sets
    categories = ["hr", "lr"] 

    # Target image sizes
    hr_landscape_size = (300, 224)  # HR Landscape format
    hr_portrait_size = (224, 300)  # HR Portrait format

    lr_landscape_size = (150, 112)  # LR Landscape format
    lr_portrait_size = (112, 150)  # LR Portrait format

    for split in splits:  # Train/Test
        for category in categories:  # HR/LR
            folder_path = os.path.join(output_folder, split, category)
            
            # Create subfolders for landscape and portrait images
            landscape_folder = os.path.join(folder_path, "landscape")
            portrait_folder = os.path.join(folder_path, "portrait")
            os.makedirs(landscape_folder, exist_ok=True)
            os.makedirs(portrait_folder, exist_ok=True)

            # Iterate through all images in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # Skip if it's already sorted into subfolders
                if os.path.isdir(file_path):
                    continue

                try:
                    # Read image to get its dimensions
                    img = cv2.imread(file_path)
                    if img is None:
                        print(f"Skipping invalid file: {filename}")
                        continue

                    height, width, _ = img.shape

                    # Classify based on HR or LR image size
                    if category == "hr":
                        if (width, height) == hr_landscape_size:
                            dest_folder = landscape_folder
                        elif (width, height) == hr_portrait_size:
                            dest_folder = portrait_folder
                        else:
                            print(f"Skipping {filename}: Unexpected HR size {width}x{height}")
                            continue  # Skip images with unexpected sizes
                    else:  # category == "lr"
                        if (width, height) == lr_landscape_size:
                            dest_folder = landscape_folder
                        elif (width, height) == lr_portrait_size:
                            dest_folder = portrait_folder
                        else:
                            print(f"Skipping {filename}: Unexpected LR size {width}x{height}")
                            continue  # Skip images with unexpected sizes

                    # Move the image to the correct subfolder
                    shutil.move(file_path, os.path.join(dest_folder, filename))
                    print(f"Moved {filename} to {dest_folder}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

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


# Run the classification
classify_and_move_images()
print("✅ Dataset has been organized into 'landscape' and 'portrait' subfolders.")