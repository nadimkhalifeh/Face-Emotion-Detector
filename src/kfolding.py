import os
import pickle
from sklearn.model_selection import KFold

def is_valid_image(file_name):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return file_name.lower().endswith(valid_extensions) and not file_name.startswith('.')

def prepare_kfold_data():
    dataset_dir = '../dataset-cleaned/'
    image_paths = []
    labels = []

    # Collect all image paths and labels
    for class_index, class_name in enumerate(['angry', 'engaged', 'happy', 'neutral']):
        class_folder = os.path.join(dataset_dir, class_name)
        for file_name in os.listdir(class_folder):
            if is_valid_image(file_name):  # Filter invalid files
                image_paths.append(os.path.join(class_folder, file_name))
                labels.append(class_index)

    return image_paths, labels


if __name__ == '__main__':
    # Prepare dataset
    image_list, label_list = prepare_kfold_data()

    # Save full dataset
    dataset = {'images': image_list, 'labels': label_list}
    with open('../full_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    # Generate K-Fold splits
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_indices = list(kfold.split(image_list))

    # Save K-Fold splits
    with open('../kfold_dataset.pkl', 'wb') as f:
        pickle.dump(fold_indices, f)

    print("K-Fold data preparation complete.")