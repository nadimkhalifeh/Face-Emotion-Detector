# This script prepares the dataset for K-Fold cross-validation. It generates two .pkl files:
# 1. One containing the entire dataset with image paths and labels.
# 2. Another containing the indices for the K-Fold splits.

import os
import pickle
from sklearn.model_selection import KFold

def prepare_data():
    dataset_dir = '../dataset-cleaned/'
    image_paths = []
    labels = []

    # Collect all image paths and their respective class labels
    for class_index, class_name in enumerate(['angry', 'engaged', 'happy', 'neutral']):
        class_folder = os.path.join(dataset_dir, class_name)
        files = os.listdir(class_folder)
        for file in files:
            image_paths.append(os.path.join(class_folder, file))
            labels.append(class_index)

    return image_paths, labels


if __name__ == '__main__':
    # Load image paths and labels
    image_list, label_list = prepare_data()

    dataset_content = {
        'images': image_list,
        'labels': label_list,
    }

    # Save the entire dataset to a pickle file
    with open('../full_dataset.pkl', 'wb') as file:
        pickle.dump(dataset_content, file)

    # Configure K-Fold cross-validation
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_indices = list(kfold.split(image_list))

    # Save the fold indices to another pickle file
    with open('../kfold_dataset.pkl', 'wb') as file:
        pickle.dump(fold_indices, file)