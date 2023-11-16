import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tarfile
import json
import numpy as np
import pandas as pd

from pathlib import Path

from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score

def move_non_image_folders(data_dir):
    """
    Move non-image folders from data_dir to a separate directory.

    :param data_dir: Path to the directory containing image folders.
    """
    # Define the directory to move non-image folders to
    parent_dir = Path(data_dir).parent
    non_image_dir = os.path.join(parent_dir, 'non_image_folders')

    # Create the directory if it doesn't exist
    if not os.path.exists(non_image_dir):
        os.makedirs(non_image_dir)

    # Iterate through each item in data_dir
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)

        if os.path.isdir(item_path):
            contains_images = any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')) for file in os.listdir(item_path))
            if not contains_images:
                destination = os.path.join(non_image_dir, item)
                shutil.move(item_path, destination)
                print(f"Moved non-image folder: {item} to {non_image_dir}")
                
def extract_model_dir(model_dir):
    print("Extracting model_dir...")
    with tarfile.open(Path(model_dir) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_dir))
    print("extracted tar.")

def get_model(model_dir):    
    print("Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(Path(model_dir) / "data_classes.txt", "r") as f:
        num_classes = len(f.readlines())
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    with open(os.path.join(model_dir, 'model.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def get_dataset(data_dir, model_dir, debug=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    with open(Path(model_dir) / "data_classes.txt", "r") as f:
        classes = [cls.strip() for cls in f.readlines()]
    test_dataset.classes = classes
    test_dataset.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    if debug: 
        # Define the size of the subset
        subset_size = 32
        indices = list(range(subset_size))
        test_dataset = Subset(test_dataset, indices)
        
    return test_dataset

def write_evaluation_baseline_csv(output_dir, all_preds, all_labels, all_confs):
    print("Writing baseline csv...")
    df = pd.DataFrame({
        'Predicted': all_preds,
        'Label': all_labels,
        'Confidence': all_confs
    })

    output_file_path = os.path.join(output_dir, 'evaluation_baseline.csv')
    df.to_csv(output_file_path, index=False)
    print(f"Wrote {output_file_path}")
    return output_file_path

def evaluate(model, test_loader, output_dir):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    
    # Prepare to store predictions and actual labels
    all_preds = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, 1)
            confs, predicted = torch.max(probs, 1)
            
            # Append predictions and labels for calculating metrics later
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                        
    # Convert to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = correct / total    
    # Calculate precision, recall
    precision = precision_score(all_labels, all_preds, average='weighted')  # Use 'micro' or 'weighted' as needed
    recall = recall_score(all_labels, all_preds, average='weighted')  # Use 'micro' or 'weighted' as needed
    
    print(f'Test Accuracy: {accuracy * 100}%')
    print(f'Test weighted precision: {precision}')
    print(f'Test weighted recall: {recall}')
    
    evaluation_report = {
        "metrics": {
            "accuracy": {
                "value": accuracy
            },
            "weighted_precision": {
                "value": precision
            },
            "weighted_recall": {
                "value": recall
            }
        },
    }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))
    print("Written evaluation.json .")
          
    write_evaluation_baseline_csv(output_dir, all_preds, all_labels, all_confs)
    
    return accuracy

def main(model_dir, data_dir, output_dir, debug=False):
    # move non image folders out
    move_non_image_folders(data_dir)
    extract_model_dir(model_dir)
    
    model = get_model(model_dir)
    test_dataset = get_dataset(data_dir, model_dir, debug)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    accuracy = evaluate(model, test_loader, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/evaluation')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    main(args.model_dir, args.data_dir, args.output_dir, debug=args.debug)
