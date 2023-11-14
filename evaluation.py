import os
import shutil
import argparse
import torch
import torch.nn as nn
import tarfile
import json
from pathlib import Path

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

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

def get_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def model_fn(model_dir, context):
    print("Loading model...")
    print("Context: ", context)
    with tarfile.open(Path(model_dir) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_dir))
    print("extracted tar ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 133
    model = get_model(num_classes)
    with open(os.path.join(model_dir, 'model.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    return model

def evaluate(model, test_loader, output_dir):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100}%')
    
    evaluation_report = {
        "metrics": {
            "accuracy": {
                "value": accuracy
            },
        },
    }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))
        
    return accuracy

def main(model_dir, data_dir, output_dir):
    # move non image folders out
    move_non_image_folders(data_dir)
    
    context = "Evaluation"
    model = model_fn(model_dir, context)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    accuracy = evaluate(model, test_loader, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/evaluation')
    args = parser.parse_args()

    main(args.model_dir, args.data_dir, args.output_dir)
