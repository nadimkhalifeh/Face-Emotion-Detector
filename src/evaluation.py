import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
from variant1 import MultiLayerFCNet  # Importing the model architecture

from sklearn.model_selection import train_test_split


class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.transform_pipeline = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for grayscale images
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load and process the image
        image = self.transform_pipeline(Image.open(self.image_paths[index]).convert('L'))
        label = self.labels[index]
        return image, label


def load_dataset():
    dataset_dir = '../dataset/'
    image_paths = []
    labels = []

    # Loop through each class folder
    for class_index, class_name in enumerate(['angry', 'engaged', 'happy', 'neutral']):
        class_dir = os.path.join(dataset_dir, class_name)
        files = os.listdir(class_dir)
        for file in files:
            labels.append(class_index)
            image_paths.append(os.path.join(class_dir, file))

    # Create a DataLoader for the dataset
    dataset = EmotionDataset(image_paths, labels)
    return DataLoader(dataset, shuffle=True, num_workers=8, drop_last=True)


def load_single_image(image_path, class_index):
    image_paths = [image_path]
    labels = [class_index]

    # Create a DataLoader for the single image
    dataset = EmotionDataset(image_paths, labels)
    return DataLoader(dataset, shuffle=False, num_workers=8, drop_last=True)


if __name__ == '__main__':
    input_dim = 64 * 64  # Input size for a 64x64 grayscale image
    hidden_dim = 50  # Number of hidden neurons
    num_classes = 4  # Four emotion classes: angry, engaged, happy, neutral
    output_message = ""

    user_choice = input("Enter \"dataset\" to evaluate the entire dataset or \"single\" to evaluate a single image: ")

    if user_choice == "dataset":
        test_loader = load_dataset()
    elif user_choice == "single":
        class_name = input("Enter the emotion category of your image (angry, engaged, happy, neutral): ")
        if class_name == 'angry':
            class_idx = 0
        elif class_name == 'engaged':
            class_idx = 1
        elif class_name == 'happy':
            class_idx = 2
        elif class_name == 'neutral':
            class_idx = 3
        else:
            raise ValueError("Invalid class name. Please choose from angry, engaged, happy, neutral.")

        image_path = input("Enter the file path of the image: ")
        test_loader = load_single_image(image_path, class_idx)

    # Setup the device for computation
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiLayerFCNet(input_dim, hidden_dim, num_classes)
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load('../models/best_model_variant1.pth'))
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted_labels = torch.max(predictions.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted_labels == labels).sum().item()

            if user_choice == "single":
                for i in range(len(images)):
                    emotion_class = ["angry", "engaged", "happy", "neutral"][predicted_labels[i].item()]
                    print(f"Predicted class: {emotion_class}")
                output_message = f"True label for the image: {class_name}"

    accuracy = 100 * total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(output_message)