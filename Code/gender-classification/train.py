import argparse
import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

# Custom dataset for subject-based splitting
class SubjectDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: list of tuples (image_path, label)
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_subject_split(data_dir, test_subjects_per_class=5, allowed_extensions=['.jpg', '.jpeg', '.png']):
    """
    For each class folder in data_dir, randomly select a fixed number of subject subfolders for testing.
    Returns:
      - train_list: list of (image_path, label) for training
      - test_list: list of (image_path, label) for testing
      - classes: list of class names
    """
    train_list = []
    test_list = []
    
    # List class names sorted alphabetically
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        subjects = sorted([d for d in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, d))])
        random.shuffle(subjects)
        test_subjects = subjects[:test_subjects_per_class]
        train_subjects = subjects[test_subjects_per_class:]
        
        def add_images(subject_list, target_list):
            for subject in subject_list:
                subject_path = os.path.join(cls_path, subject)
                for root, _, files in os.walk(subject_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in allowed_extensions):
                            target_list.append((os.path.join(root, file), class_to_idx[cls]))
        
        add_images(train_subjects, train_list)
        add_images(test_subjects, test_list)
    
    return train_list, test_list, classes

def get_model(model_name, num_classes, local_weights_path=None):
    """
    Load a model architecture. If local_weights_path is provided, the checkpoint is loaded—
    which can be either a state dictionary or a complete model. If it is a complete model,
    the classifier layer is adjusted if needed to match num_classes.
    If no local weights are provided, the default pretrained weights (from torchvision) are used,
    and the final classification layer is replaced accordingly.
    """
    if local_weights_path is not None:
        # Load checkpoint (complete model or state_dict)
        checkpoint = torch.load(local_weights_path, weights_only=False)
        if isinstance(checkpoint, dict):
            # Checkpoint is a state dictionary.
            if model_name == 'resnet18':
                model = models.resnet18(pretrained=False)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
                model.load_state_dict(checkpoint)
            elif model_name == 'vgg16':
                model = models.vgg16(pretrained=False)
                in_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(in_features, num_classes)
                model.load_state_dict(checkpoint)
            elif model_name == 'densenet121':
                model = models.densenet121(pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
                model.load_state_dict(checkpoint)
            elif model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=False)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, num_classes)
                model.load_state_dict(checkpoint)
            elif model_name == 'mobilenet_v2':
                model = models.mobilenet_v2(pretrained=False)
                in_features = model.classifier[1].in_features
                # Replace classifier to output the correct number of classes
                model.classifier[1] = nn.Linear(in_features, num_classes)
                with torch.serialization.safe_globals([models.mobilenetv2.MobileNetV2]):
                    model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Local weights loading is not implemented for model '{model_name}'.")
        else:
            # Checkpoint is a complete model.
            model = checkpoint
            # For safety, update the classifier if the output shape doesn't match.
            if model_name == 'resnet18':
                in_features = model.fc.in_features
                if model.fc.out_features != num_classes:
                    model.fc = nn.Linear(in_features, num_classes)
            elif model_name == 'vgg16':
                in_features = model.classifier[6].in_features
                if model.classifier[6].out_features != num_classes:
                    model.classifier[6] = nn.Linear(in_features, num_classes)
            elif model_name == 'densenet121':
                in_features = model.classifier.in_features
                if model.classifier.out_features != num_classes:
                    model.classifier = nn.Linear(in_features, num_classes)
            elif model_name == 'efficientnet_b0':
                in_features = model.classifier[1].in_features
                if model.classifier[1].out_features != num_classes:
                    model.classifier[1] = nn.Linear(in_features, num_classes)
            elif model_name == 'mobilenet_v2':
                in_features = model.classifier[1].in_features
                if model.classifier[1].out_features != num_classes:
                    model.classifier[1] = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Local weights loading is not implemented for model '{model_name}'.")
        return model
    else:
        # No local weights provided; use default pretrained weights.
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == 'yolov9':
            raise NotImplementedError("YOLOv9 is not directly supported as a classification model. "
                                      "Please refer to a dedicated YOLO implementation (e.g., Ultralytics YOLO) "
                                      "and adjust the training script accordingly.")
        else:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from "
                             "'resnet18', 'vgg16', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', or 'yolov9'.")
        return model

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}] Average Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Training Script with TensorBoard, Subject-Based Splitting, '
                    'Data Augmentation, Regularization, and LR Decay'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the dataset directory (expects class folders with subject subfolders)')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'vgg16', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', 'yolov9'],
                        help='Choose the model architecture')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 regularization weight decay')
    parser.add_argument('--test-subjects', type=int, default=5,
                        help='Number of subjects to reserve for testing per class')
    parser.add_argument('--subject-based', action='store_true',
                        help='Use subject-based splitting instead of pre-split ImageFolder structure')
    parser.add_argument('--local-weights', type=str, default=None,
                        help='Path to the local pretrained weights file')
    args = parser.parse_args()

    # Set up TensorBoard writer (logs saved in "runs")
    writer = SummaryWriter()

    # Define transforms for training (with augmentation) and testing (without augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create datasets: if subject_based flag is set, perform a subject-based split.
    if args.subject_based:
        train_list, test_list, classes = prepare_subject_split(args.data_dir, test_subjects_per_class=args.test_subjects)
        train_dataset = SubjectDataset(train_list, transform=train_transform)
        test_dataset = SubjectDataset(test_list, transform=test_transform)
    else:
        # Assumes a pre-split dataset with "train" and "val" subdirectories.
        from torchvision import datasets
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=train_transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=test_transform)
        classes = train_dataset.classes

    print(f"Detected classes: {classes}")
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the model
    try:
        model = get_model(args.model, num_classes, local_weights_path=args.local_weights)
    except NotImplementedError as e:
        print(e)
        return
    model = model.to(device)
    
    # Set up loss function and optimizer with weight decay for regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set up a learning rate scheduler that decays LR every 8 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    # Create directory to save models
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    best_accuracy = 0.0

    # Training and validation loop
    for epoch in range(1, args.epochs + 1):
        print(f"Learning Rate before epoch {epoch}: {optimizer.param_groups[0]['lr']:.8f}")

        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, accuracy = validate(model, device, test_loader, criterion)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        
        # Step the scheduler
        scheduler.step()
        print(f"Learning Rate after epoch {epoch}: {optimizer.param_groups[0]['lr']:.6f}")

        # Save model for current epoch
        epoch_state_path = os.path.join(model_dir, f"{args.model}_epoch{epoch}_model_state.pth")
        epoch_complete_path = os.path.join(model_dir, f"{args.model}_epoch{epoch}_complete_model.pth")
        torch.save(model.state_dict(), epoch_state_path)
        torch.save(model, epoch_complete_path)
        print(f"Saved model for epoch {epoch}.")

        # Update best model if current validation accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_path = os.path.join(model_dir, "best_model_state.pth")
            best_complete_path = os.path.join(model_dir, "best_model_complete.pth")
            torch.save(model.state_dict(), best_state_path)
            torch.save(model, best_complete_path)
            print(f"New best model at epoch {epoch} with accuracy {accuracy:.2f}%.")

    writer.close()
    print(f"Training complete. All models saved in '{model_dir}'. Best model achieved {best_accuracy:.2f}% accuracy.")

if __name__ == '__main__':
    main()

# import argparse
# import os
# import random
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torch.utils.tensorboard import SummaryWriter

# # Custom dataset for subject-based splitting
# class SubjectDataset(Dataset):
#     def __init__(self, data_list, transform=None):
#         """
#         data_list: list of tuples (image_path, label)
#         """
#         self.data_list = data_list
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, idx):
#         img_path, label = self.data_list[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# def prepare_subject_split(data_dir, test_subjects_per_class=5, allowed_extensions=['.jpg', '.jpeg', '.png']):
#     """
#     For each class folder in data_dir, randomly select a fixed number of subject subfolders for testing.
#     Returns:
#       - train_list: list of (image_path, label) for training
#       - test_list: list of (image_path, label) for testing
#       - classes: list of class names
#     """
#     train_list = []
#     test_list = []
    
#     # List class names sorted alphabetically
#     classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
#     class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
#     for cls in classes:
#         cls_path = os.path.join(data_dir, cls)
#         subjects = sorted([d for d in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, d))])
#         random.shuffle(subjects)
#         test_subjects = subjects[:test_subjects_per_class]
#         train_subjects = subjects[test_subjects_per_class:]
        
#         def add_images(subject_list, target_list):
#             for subject in subject_list:
#                 subject_path = os.path.join(cls_path, subject)
#                 for root, _, files in os.walk(subject_path):
#                     for file in files:
#                         if any(file.lower().endswith(ext) for ext in allowed_extensions):
#                             target_list.append((os.path.join(root, file), class_to_idx[cls]))
        
#         add_images(train_subjects, train_list)
#         add_images(test_subjects, test_list)
    
#     return train_list, test_list, classes

# def get_model(model_name, num_classes):
#     """
#     Load a pretrained model from torchvision (or a placeholder for YOLOv9)
#     and replace the final classification layer to match the number of classes.
#     """
#     if model_name == 'resnet18':
#         model = models.resnet18(pretrained=True)
#         in_features = model.fc.in_features
#         model.fc = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'vgg16':
#         model = models.vgg16(pretrained=True)
#         in_features = model.classifier[6].in_features
#         model.classifier[6] = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'densenet121':
#         model = models.densenet121(pretrained=True)
#         in_features = model.classifier.in_features
#         model.classifier = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'efficientnet_b0':
#         model = models.efficientnet_b0(pretrained=True)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'mobilenet_v2':
#         model = models.mobilenet_v2(pretrained=True)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'yolov9':
#         raise NotImplementedError("YOLOv9 is not directly supported as a classification model. "
#                                   "Please refer to a dedicated YOLO implementation (e.g., Ultralytics YOLO) "
#                                   "and adjust the training script accordingly.")
#     else:
#         raise ValueError(f"Model '{model_name}' is not supported. Choose from "
#                          "'resnet18', 'vgg16', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', or 'yolov9'.")
#     return model

# def train(model, device, train_loader, optimizer, criterion, epoch):
#     model.train()
#     running_loss = 0.0
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         data, targets = data.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if batch_idx % 10 == 0:
#             print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

#     avg_loss = running_loss / len(train_loader)
#     print(f"Epoch [{epoch}] Average Training Loss: {avg_loss:.4f}")
#     return avg_loss

# def validate(model, device, val_loader, criterion):
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, targets in val_loader:
#             data, targets = data.to(device), targets.to(device)
#             outputs = model(data)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#     avg_loss = val_loss / len(val_loader)
#     accuracy = 100. * correct / total
#     print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
#     return avg_loss, accuracy

# def main():
#     parser = argparse.ArgumentParser(
#         description='PyTorch Classification Training Script with TensorBoard, Subject-Based Splitting, '
#                     'Data Augmentation, Regularization, and LR Decay'
#     )
#     parser.add_argument('--data-dir', type=str, required=True,
#                         help='Path to the dataset directory (expects class folders with subject subfolders)')
#     parser.add_argument('--model', type=str, default='resnet18',
#                         choices=['resnet18', 'vgg16', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', 'yolov9'],
#                         help='Choose the model architecture')
#     parser.add_argument('--epochs', type=int, default=10,
#                         help='Number of training epochs')
#     parser.add_argument('--batch-size', type=int, default=32,
#                         help='Input batch size for training')
#     parser.add_argument('--lr', type=float, default=0.001,
#                         help='Learning rate')
#     parser.add_argument('--num-workers', type=int, default=4,
#                         help='Number of workers for data loading')
#     parser.add_argument('--weight-decay', type=float, default=1e-4,
#                         help='L2 regularization weight decay')
#     parser.add_argument('--test-subjects', type=int, default=5,
#                         help='Number of subjects to reserve for testing per class')
#     parser.add_argument('--subject-based', action='store_true',
#                         help='Use subject-based splitting instead of pre-split ImageFolder structure')
#     args = parser.parse_args()

#     # Set up TensorBoard writer (logs saved in "runs")
#     writer = SummaryWriter()

#     # Define transforms for training (with augmentation) and testing (without augmentation)
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # Create datasets: if subject_based flag is set, perform a subject-based split.
#     if args.subject_based:
#         train_list, test_list, classes = prepare_subject_split(args.data_dir, test_subjects_per_class=args.test_subjects)
#         train_dataset = SubjectDataset(train_list, transform=train_transform)
#         test_dataset = SubjectDataset(test_list, transform=test_transform)
#     else:
#         # Assumes a pre-split dataset with "train" and "val" subdirectories.
#         from torchvision import datasets
#         train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=train_transform)
#         test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=test_transform)
#         classes = train_dataset.classes

#     print(f"Detected classes: {classes}")
#     num_classes = len(classes)
#     print(f"Number of classes: {num_classes}")

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Create the model
#     try:
#         model = get_model(args.model, num_classes)
#     except NotImplementedError as e:
#         print(e)
#         return
#     model = model.to(device)
    
#     # Set up loss function and optimizer with weight decay for regularization
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
#     # Set up a learning rate scheduler that decays LR every 5 epochs
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
#      # Create directory to save models
#     model_dir = "saved_models"
#     os.makedirs(model_dir, exist_ok=True)
#     best_accuracy = 0.0

#     # Training and validation loop
#     for epoch in range(1, args.epochs + 1):
#         print(f"Learning Rate after epoch {epoch}: {optimizer.param_groups[0]['lr']:.8f}")

#         train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
#         val_loss, accuracy = validate(model, device, test_loader, criterion)

#         # Log metrics to TensorBoard
#         writer.add_scalar('Loss/Train', train_loss, epoch)
#         writer.add_scalar('Loss/Validation', val_loss, epoch)
#         writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        
#         # Step the scheduler to decay the learning rate every 5 epochs
#         scheduler.step()
#         print(f"Learning Rate after epoch {epoch}: {optimizer.param_groups[0]['lr']:.6f}")

#     # Save the trained model state dictionary and the complete model
#     # Save the model for the current epoch
#         epoch_state_path = os.path.join(model_dir, f"{args.model}_epoch{epoch}_model_state.pth")
#         epoch_complete_path = os.path.join(model_dir, f"{args.model}_epoch{epoch}_complete_model.pth")
#         torch.save(model.state_dict(), epoch_state_path)
#         torch.save(model, epoch_complete_path)
#         print(f"Saved model for epoch {epoch}.")

#         # Update best model if current validation accuracy is higher
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_state_path = os.path.join(model_dir, "best_model_state.pth")
#             best_complete_path = os.path.join(model_dir, "best_model_complete.pth")
#             torch.save(model.state_dict(), best_state_path)
#             torch.save(model, best_complete_path)
#             print(f"New best model at epoch {epoch} with accuracy {accuracy:.2f}%.")
#     # state_path = f"saved_model/{args.model}_model_state.pth"
#     # complete_path = f"saved_model/{args.model}_complete_model.pth"
#     # torch.save(model.state_dict(), state_path)
#     # torch.save(model, complete_path)
#     writer.close()
#     print(f"Training complete. All models saved in '{model_dir}'. Best model achieved {best_accuracy:.2f}% accuracy.")

# if __name__ == '__main__':
#     main()





# import argparse
# import os
# import random
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torch.utils.tensorboard import SummaryWriter

# # Custom dataset for subject-based splitting
# class SubjectDataset(Dataset):
#     def __init__(self, data_list, transform=None):
#         """
#         data_list: list of tuples (image_path, label)
#         """
#         self.data_list = data_list
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, idx):
#         img_path, label = self.data_list[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# def prepare_subject_split(data_dir, test_subjects_per_class=5, allowed_extensions=['.jpg', '.jpeg', '.png']):
#     """
#     For each class folder in data_dir, randomly select a fixed number of subject subfolders for testing.
#     Returns:
#       - train_list: list of (image_path, label) for training
#       - test_list: list of (image_path, label) for testing
#       - classes: list of class names
#     """
#     train_list = []
#     test_list = []
    
#     # List class names sorted alphabetically
#     classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
#     class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
#     for cls in classes:
#         cls_path = os.path.join(data_dir, cls)
#         subjects = sorted([d for d in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, d))])
#         random.shuffle(subjects)
#         test_subjects = subjects[:test_subjects_per_class]
#         train_subjects = subjects[test_subjects_per_class:]
        
#         def add_images(subject_list, target_list):
#             for subject in subject_list:
#                 subject_path = os.path.join(cls_path, subject)
#                 for root, _, files in os.walk(subject_path):
#                     for file in files:
#                         if any(file.lower().endswith(ext) for ext in allowed_extensions):
#                             target_list.append((os.path.join(root, file), class_to_idx[cls]))
        
#         add_images(train_subjects, train_list)
#         add_images(test_subjects, test_list)
    
#     return train_list, test_list, classes

# def get_model(model_name, num_classes):
#     """
#     Load a pretrained model from torchvision (or a placeholder for YOLOv9)
#     and replace the final classification layer to match the number of classes.
#     """
#     if model_name == 'resnet18':
#         model = models.resnet18(pretrained=True)
#         in_features = model.fc.in_features
#         model.fc = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'vgg16':
#         model = models.vgg16(pretrained=True)
#         in_features = model.classifier[6].in_features
#         model.classifier[6] = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'densenet121':
#         model = models.densenet121(pretrained=True)
#         in_features = model.classifier.in_features
#         model.classifier = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'efficientnet_b0':
#         model = models.efficientnet_b0(pretrained=True)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'mobilenet_v2':
#         model = models.mobilenet_v2(pretrained=True)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, num_classes)
    
#     elif model_name == 'yolov9':
#         raise NotImplementedError("YOLOv9 is not directly supported as a classification model. "
#                                   "Please refer to a dedicated YOLO implementation (e.g., Ultralytics YOLO) "
#                                   "and adjust the training script accordingly.")
#     else:
#         raise ValueError(f"Model '{model_name}' is not supported. Choose from "
#                          "'resnet18', 'vgg16', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', or 'yolov9'.")
#     return model

# def train(model, device, train_loader, optimizer, criterion, epoch):
#     model.train()
#     running_loss = 0.0
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         data, targets = data.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if batch_idx % 10 == 0:
#             print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

#     avg_loss = running_loss / len(train_loader)
#     print(f"Epoch [{epoch}] Average Training Loss: {avg_loss:.4f}")
#     return avg_loss

# def validate(model, device, val_loader, criterion):
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, targets in val_loader:
#             data, targets = data.to(device), targets.to(device)
#             outputs = model(data)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#     avg_loss = val_loss / len(val_loader)
#     accuracy = 100. * correct / total
#     print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
#     return avg_loss, accuracy

# def main():
#     parser = argparse.ArgumentParser(
#         description='PyTorch Classification Training Script with TensorBoard, Subject-Based Splitting, '
#                     'Data Augmentation, Regularization, and LR Decay'
#     )
#     parser.add_argument('--data-dir', type=str, required=True,
#                         help='Path to the dataset directory (expects class folders with subject subfolders)')
#     parser.add_argument('--model', type=str, default='resnet18',
#                         choices=['resnet18', 'vgg16', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', 'yolov9'],
#                         help='Choose the model architecture')
#     parser.add_argument('--epochs', type=int, default=10,
#                         help='Number of training epochs')
#     parser.add_argument('--batch-size', type=int, default=32,
#                         help='Input batch size for training')
#     parser.add_argument('--lr', type=float, default=0.001,
#                         help='Learning rate')
#     parser.add_argument('--num-workers', type=int, default=4,
#                         help='Number of workers for data loading')
#     parser.add_argument('--weight-decay', type=float, default=1e-4,
#                         help='L2 regularization weight decay')
#     parser.add_argument('--test-subjects', type=int, default=5,
#                         help='Number of subjects to reserve for testing per class')
#     parser.add_argument('--subject-based', action='store_true',
#                         help='Use subject-based splitting instead of pre-split ImageFolder structure')
#     args = parser.parse_args()

#     # Set up TensorBoard writer (logs saved in "runs")
#     writer = SummaryWriter()

#     # Define transforms for training (with augmentation) and testing (without augmentation)
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(512),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#     test_transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # Create datasets: if subject-based flag is set, perform a subject-based split.
#     if args.subject_based:
#         train_list, test_list, classes = prepare_subject_split(args.data_dir, test_subjects_per_class=args.test_subjects)
#         train_dataset = SubjectDataset(train_list, transform=train_transform)
#         test_dataset = SubjectDataset(test_list, transform=test_transform)
#     else:
#         # Assumes a pre-split dataset with "train" and "val" subdirectories.
#         from torchvision import datasets
#         train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=train_transform)
#         test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=test_transform)
#         classes = train_dataset.classes

#     print(f"Detected classes: {classes}")
#     num_classes = len(classes)
#     print(f"Number of classes: {num_classes}")

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Create the model
#     try:
#         model = get_model(args.model, num_classes)
#     except NotImplementedError as e:
#         print(e)
#         return
#     model = model.to(device)
    
#     # Set up loss function and optimizer with weight decay for regularization
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
#     # Set up a learning rate scheduler that decays LR every 5 epochs
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
#     # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
#     # Create directory to save models
#     model_dir = "saved_models"
#     os.makedirs(model_dir, exist_ok=True)
#     best_accuracy = 0.0

#     # Training and validation loop
#     for epoch in range(1, args.epochs + 1):
#         train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
#         val_loss, accuracy = validate(model, device, test_loader, criterion)

#         # Log metrics to TensorBoard
#         writer.add_scalar('Loss/Train', train_loss, epoch)
#         writer.add_scalar('Loss/Validation', val_loss, epoch)
#         writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        
#         # Step the scheduler to decay the learning rate every 5 epochs
#         scheduler.step()
#         print(f"Learning Rate after epoch {epoch}: {optimizer.param_groups[0]['lr']:.6f}")
        
#         # Save the model for the current epoch
#         epoch_state_path = os.path.join(model_dir, f"{args.model}_epoch{epoch}_model_state.pth")
#         epoch_complete_path = os.path.join(model_dir, f"{args.model}_epoch{epoch}_complete_model.pth")
#         torch.save(model.state_dict(), epoch_state_path)
#         torch.save(model, epoch_complete_path)
#         print(f"Saved model for epoch {epoch}.")

#         # Update best model if current validation accuracy is higher
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_state_path = os.path.join(model_dir, "best_model_state.pth")
#             best_complete_path = os.path.join(model_dir, "best_model_complete.pth")
#             torch.save(model.state_dict(), best_state_path)
#             torch.save(model, best_complete_path)
#             print(f"New best model at epoch {epoch} with accuracy {accuracy:.2f}%.")

#     writer.close()
#     print(f"Training complete. All models saved in '{model_dir}'. Best model achieved {best_accuracy:.2f}% accuracy.")

# if __name__ == '__main__':
#     main()
