# This script provides data visualization for image datasets. It generates:
# 1. A bar chart to display the number of images in each category.
# 2. Histograms illustrating the aggregated pixel intensities for each category.
# 3. Pixel intensity histograms for 15 randomly chosen images per category.

import os
import matplotlib.pyplot as plt
from PIL import Image
import random


# Function to create a bar chart showing the distribution of images across categories
def visualize_class_distribution(dataset):
    categories = ['angry', 'happy', 'engaged', 'neutral']
    # Paths to the respective category folders
    category_paths = [f'../dataset-cleaned/{label}' for label in categories]

    # Count the number of images in each category
    image_counts = [len(os.listdir(path)) for path in category_paths]

    # Plot the bar chart
    plt.figure()
    plt.bar(categories, image_counts, color='skyblue')
    plt.xlabel('Image Categories')
    plt.ylabel('Count of Images')
    plt.title(f'Image Distribution by Category in {dataset} Dataset')
    plt.show()


# Function to generate histograms for pixel intensities across categories
def visualize_pixel_intensity_distribution(dataset):
    categories = ['angry', 'happy', 'engaged', 'neutral']
    colors = ['red', 'green', 'blue', 'gray']  # Colors for each histogram
    category_paths = [f'../dataset-cleaned/{label}' for label in categories]

    plt.figure(figsize=(12, 12))

    for idx, path in enumerate(category_paths):
        pixel_values = []

        # Aggregate pixel intensities for all images in the category
        for filename in os.listdir(path):
            img = Image.open(os.path.join(path, filename))

            # Ensure image is in grayscale
            if img.mode != 'L':
                img = img.convert('L')

            pixel_values.extend(list(img.getdata()))

        # Plot the histogram for the current category
        plt.subplot(2, 2, idx + 1)
        plt.hist(pixel_values, bins=256, color=colors[idx], alpha=0.7)
        plt.title(f'Pixel Intensity: {categories[idx]} ({dataset})')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Function to display histograms for 15 randomly sampled images from each category
def visualize_sample_images(dataset):
    categories = ['angry', 'happy', 'engaged', 'neutral']
    category_paths = [f'../dataset-cleaned/{label}' for label in categories]

    for cat_idx, path in enumerate(category_paths):
        fig, axes = plt.subplots(5, 6, figsize=(20, 12))
        axes = axes.flatten()

        # Select 15 random images
        image_files = os.listdir(path)
        sampled_images = random.sample(image_files, 15)

        for i, img_name in enumerate(sampled_images):
            img = Image.open(os.path.join(path, img_name))
            histogram = img.histogram()

            # Plot the image and its histogram
            axes[2 * i].imshow(img, cmap='gray')
            axes[2 * i].axis('off')
            axes[2 * i + 1].bar(range(256), histogram, color='black', alpha=0.7)

        fig.suptitle(f'Pixel Intensity Histograms for {categories[cat_idx]} Images ({dataset} Dataset)', fontsize=16)
        plt.tight_layout()
        plt.show()


# Visualizations for the training dataset
visualize_class_distribution('train')
visualize_pixel_intensity_distribution('train')
visualize_sample_images('train')