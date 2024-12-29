# This script processes a dataset of images to ensure uniformity for training and evaluation purposes
# The script standardizes image size, naming conventions, sharpness, contrast, and brightness.

import os
import shutil
from PIL import Image, ImageEnhance, ImageStat


def calculate_brightness(img):
    stats = ImageStat.Stat(img)
    return stats.mean[0]


def calculate_contrast(img):
    stats = ImageStat.Stat(img)
    return stats.stddev[0]


def calculate_sharpness(img):
    stats = ImageStat.Stat(img)
    return stats.stddev[0]


def is_brightness_high(img, threshold=150):
    brightness = calculate_brightness(img)
    return brightness > threshold, brightness


def is_brightness_low(img, threshold=100):
    brightness = calculate_brightness(img)
    return brightness < threshold, brightness


def is_contrast_high(img, threshold=175):
    contrast = calculate_contrast(img)
    return contrast > threshold, contrast


def is_contrast_low(img, threshold=60):
    contrast = calculate_contrast(img)
    return contrast < threshold, contrast


def is_sharpness_high(img, threshold=150):
    sharpness = calculate_sharpness(img)
    return sharpness > threshold, sharpness


def is_sharpness_low(img, threshold=50):
    sharpness = calculate_sharpness(img)
    return sharpness < threshold, sharpness


# Function to adjust an individual image for uniformity
def process_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize image

    # Brightness adjustment
    high_brightness, brightness = is_brightness_high(img)
    low_brightness, brightness = is_brightness_low(img)
    if high_brightness:
        adjust_factor = (brightness - 150) / 150
        img = ImageEnhance.Brightness(img).enhance(1 - adjust_factor)
    elif low_brightness:
        adjust_factor = (100 - brightness) / 100
        img = ImageEnhance.Brightness(img).enhance(1 + adjust_factor)

    # Contrast adjustment
    high_contrast, contrast = is_contrast_high(img)
    low_contrast, contrast = is_contrast_low(img)
    if high_contrast:
        adjust_factor = (contrast - 175) / 175
        img = ImageEnhance.Contrast(img).enhance(1 - adjust_factor)
    elif low_contrast:
        adjust_factor = (60 - contrast) / 60
        img = ImageEnhance.Contrast(img).enhance(1 + adjust_factor)

    # Sharpness adjustment
    high_sharpness, sharpness = is_sharpness_high(img)
    low_sharpness, sharpness = is_sharpness_low(img)
    if high_sharpness:
        adjust_factor = (sharpness - 150) / 150
        img = ImageEnhance.Sharpness(img).enhance(1 - adjust_factor)
    elif low_sharpness:
        adjust_factor = (50 - sharpness) / 50
        img = ImageEnhance.Sharpness(img).enhance(1 + adjust_factor)

    return img


# Function to manage dataset cleanup and renaming
def process_dataset(input_dir, target_size=(64, 64)):
    print('Initiating dataset cleanup...')
    output_dir = os.path.join(os.path.dirname(input_dir), f"{os.path.basename(input_dir)}-cleaned")

    # Remove existing processed dataset directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    shutil.copytree(input_dir, output_dir)

    for root, subdirs, files in os.walk(output_dir):
        for subdir in subdirs:
            image_counter = 0
            subdir_path = os.path.join(root, subdir)
            parent_folder = os.path.basename(root)
            for file in os.listdir(subdir_path):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(subdir_path, file)
                    processed_image = process_image(image_path, target_size)

                    # Construct new file name
                    file_ext = os.path.splitext(file)[1]
                    new_file_name = f"{parent_folder}-{subdir}_{image_counter:04d}{file_ext}"
                    new_file_path = os.path.join(subdir_path, new_file_name)

                    # Save the adjusted image and remove the old one
                    processed_image.save(new_file_path)
                    os.remove(image_path)

                    image_counter += 1

    print('Dataset cleanup completed.')


# Set dataset directory and process
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dataset_path = os.path.join(parent_dir, 'dataset')  # Assume 'dataset' is in the parent directory

process_dataset(dataset_path)