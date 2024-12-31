import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)
from torchvision.transforms import transforms
from trainAI_main import MultiLayerFCNet as MainModel
from variant1 import MultiLayerFCNet as Variant1Model
from variant2 import MultiLayerFCNet as Variant2Model
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn


# Custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.image_paths[index]).convert('L'))
        label = self.labels[index]
        return image, label


# Function to evaluate a model's performance
def evaluate(model_checkpoint, model_class):
    # Initialize the model
    model = model_class(64 * 64, 50, 4)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)
    model.eval()

    predictions = []
    ground_truth = []

    # Evaluate model on test data
    with torch.no_grad():
        for batch, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            ground_truth.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Compute metrics
    acc = accuracy_score(ground_truth, predictions)
    detailed_report = classification_report(ground_truth, predictions, output_dict=True)
    conf_matrix = confusion_matrix(ground_truth, predictions)

    # Compute additional macro- and micro-averaged metrics
    macro_metrics = {
        'precision': precision_score(ground_truth, predictions, average='macro'),
        'recall': recall_score(ground_truth, predictions, average='macro'),
        'f1': f1_score(ground_truth, predictions, average='macro')
    }

    micro_metrics = {
        'precision': precision_score(ground_truth, predictions, average='micro'),
        'recall': recall_score(ground_truth, predictions, average='micro'),
        'f1': f1_score(ground_truth, predictions, average='micro')
    }

    detailed_report['macro avg'] = macro_metrics
    detailed_report['micro avg'] = micro_metrics

    return acc, detailed_report, conf_matrix


# Main script
if __name__ == '__main__':
    # Load test dataset
    with open('../dataset_splits.pkl', 'rb') as f:
        data_splits = pickle.load(f)

    X_test = data_splits['X_test']
    y_test = data_splits['y_test']

    # Create DataLoader for test data
    test_dataset = EmotionDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)

    # Select device for computation
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Evaluate all models
    main_acc, main_report, main_conf_matrix = evaluate('../models/best_model_main.pth', MainModel)
    variant1_acc, variant1_report, variant1_conf_matrix = evaluate('../models/best_model_variant1.pth', Variant1Model)
    variant2_acc, variant2_report, variant2_conf_matrix = evaluate('../models/best_model_variant2.pth', Variant2Model)

    # Display confusion matrices
    for model_name, matrix in zip(
        ["Main Model", "Variant 1", "Variant 2"],
        [main_conf_matrix, variant1_conf_matrix, variant2_conf_matrix]
    ):
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['angry', 'engaged', 'happy', 'neutral'])
        disp.plot()
        plt.title(f"Confusion Matrix: {model_name}")
        plt.show()

    # Summarize metrics in a table
    metrics_summary = {
        'Model': ['Main Model', 'Variant 1', 'Variant 2'],
        'P_macro': [
            main_report['macro avg']['precision'],
            variant1_report['macro avg']['precision'],
            variant2_report['macro avg']['precision']
        ],
        'R_macro': [
            main_report['macro avg']['recall'],
            variant1_report['macro avg']['recall'],
            variant2_report['macro avg']['recall']
        ],
        'F_macro': [
            main_report['macro avg']['f1'],
            variant1_report['macro avg']['f1'],
            variant2_report['macro avg']['f1']
        ],
        'P_micro': [
            main_report['micro avg']['precision'],
            variant1_report['micro avg']['precision'],
            variant2_report['micro avg']['precision']
        ],
        'R_micro': [
            main_report['micro avg']['recall'],
            variant1_report['micro avg']['recall'],
            variant2_report['micro avg']['recall']
        ],
        'F_micro': [
            main_report['micro avg']['f1'],
            variant1_report['micro avg']['f1'],
            variant2_report['micro avg']['f1']
        ],
        'Accuracy': [main_acc, variant1_acc, variant2_acc]
    }

    # Create and display the DataFrame
    metrics_df = pd.DataFrame(metrics_summary)
    print("Summary of Model Metrics:")
    print(metrics_df)