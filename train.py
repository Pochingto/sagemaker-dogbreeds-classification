import os

import argparse
import io
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

from pathlib import Path
from PIL import Image

# Import additional requirements
import boto3
from sagemaker.session import Session
from sagemaker.experiments.run import load_run

def get_dataset(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # Normalizing the data
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return dataset, train_dataset, val_dataset

def get_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Using a pretrained ResNet18 model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Adjusting the final layer for our number of classes
    
    return model
        
def model_fn(model_dir, context):
    print("Loading model...")
    print("Context: ", context)
    num_classes = 133
    model = get_model(num_classes)
    with open(os.path.join(model_dir, 'model.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def train(num_epochs, batch_size, train_dataset, val_dataset, model, model_dir):

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    print("Start training...")
    best_val_accuracy = -1.0
    
    session = Session(boto3.session.Session(region_name="us-east-1"))
    with load_run(experiment_name="DogBreedsClassification", run_name=f"run-{int(time.time())}", sagemaker_session=session) as run:
        # Define values for the parameters to log
        run.log_parameters({
            "batch_size": batch_size,
            "epochs": num_epochs
        })
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            print(f'Validation Accuracy: {100 * val_accuracy}%')
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), Path(model_dir) / 'model.pt')
                print(f'Checkpoint saved at epoch {epoch+1} with validation accuracy: {100 * val_accuracy}%')
                
            run.log_metric(name="train_loss", value=running_loss/len(train_loader), step=epoch)
            run.log_metric(name="val_acc", value=val_accuracy, step=epoch)

    print('Training complete')
    
def main(data_dir, model_dir, output_dir, num_epochs, batch_size, debug=False):
    
    dataset, train_dataset, val_dataset = get_dataset(data_dir)
    # write to data_classes.txt
    data_file_path = os.path.join(model_dir, "data_classes.txt")
    with open(data_file_path, "w") as f:
        for cls in dataset.classes:
            f.write(f"{cls}\n")
    print(f"Classes written to {data_file_path}")
    
    num_classes = len(dataset.classes)
    model = get_model(num_classes)
    
    if debug: 
        # Define the size of the subset
        subset_size = batch_size

        # Generate random indices
        indices = list(range(subset_size))
        train_dataset = Subset(train_dataset, indices)
        val_dataset = Subset(val_dataset, indices)
    
    train(num_epochs, batch_size, train_dataset, val_dataset, model, model_dir)
    
if __name__ == "__main__":
    print("Entering main func...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--output_dir", type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args, _ = parser.parse_known_args()
    
    main(args.data_dir, args.model_dir, args.output_dir, args.epochs, args.batch_size, args.debug)