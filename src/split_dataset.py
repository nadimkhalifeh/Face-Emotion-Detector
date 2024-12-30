import os
import pickle
from sklearn.model_selection import train_test_split


def is_valid_image(file_name):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return file_name.lower().endswith(valid_extensions)

def fetchData():
    path = input("Type the path of the dataset you would like to split (i.e. '../dataset-cleaned/'): ")
    allaimges = []
    clsLabel = []

    for idx, cls in enumerate(['angry', 'engaged', 'happy', 'neutral']):
        Cpath = os.path.join(path, cls)
        # Filter out invalid files and hidden files
        F = [f for f in os.listdir(Cpath) if not f.startswith('.') and is_valid_image(f)]
        for im in F:
            allaimges.append(os.path.join(Cpath, im))
            clsLabel.append(idx)

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        allaimges, clsLabel, test_size=0.30, random_state=42, stratify=clsLabel
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
if __name__ == '__main__':
    X_train, X_valid, X_test, y_train, y_valid, y_test = fetchData()
    
    splits = {
        'X_train': X_train,
        'X_valid': X_valid,
        'X_test': X_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test
    }
    
    with open('../dataset_splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
