import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_model(model_name, num_classes):
    """
    Load the specified model architecture and adjust the final classification layer.
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'yolov9':
        raise NotImplementedError("YOLOv9 is not supported for classification testing in this script.")
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Test a trained model and compute a confusion matrix")
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['resnet18', 'vgg16', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', 'yolov9'],
                        help="Model architecture to use for inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (state dictionary)")
    parser.add_argument('--num-classes', type=int, default=4, required=True,
                        help="Number of classes the model was trained on")
    parser.add_argument('--test-dir', type=str, required=True,
                        help="Path to the test dataset directory (structured in subdirectories for each class)")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for testing")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of workers for data loading")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # Define the transformation (should match your training/testing preprocessing)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the test dataset using ImageFolder (expects a folder with class subdirectories)
    test_dataset = datasets.ImageFolder(root=args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_preds = []
    all_labels = []

    # Inference loop over the test dataset
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Display confusion matrix using scikit-learn's ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    main()
