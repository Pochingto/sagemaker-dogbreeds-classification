import os
import argparse
import shutil
import random
from pathlib import Path

def split_data(data_dir, output_dir, train_ratio):
    data_path = Path(data_dir)
    train_path = Path(output_dir) / 'train'
    test_path = Path(output_dir) / 'test'
    
    # Create train and test directories
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    for class_folder in os.listdir(data_path):
        class_path = data_path / class_folder
        # check if is folder
        if not os.path.isdir(class_path):
            continue
            
        images = os.listdir(class_path)
        random.shuffle(images)
        
        split_index = int(train_ratio * len(images))
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Create class folders in train and test directories
        (train_path / class_folder).mkdir(exist_ok=True)
        (test_path / class_folder).mkdir(exist_ok=True)
        
        # Copy images to respective train/test folders
        for image in train_images:
            shutil.copy2(class_path / image, train_path / class_folder / image)

        for image in test_images:
            shutil.copy2(class_path / image, test_path / class_folder / image)

    print("Data split into training and test sets completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    args = parser.parse_args()

    split_data(args.data_dir, args.output_dir, args.train_ratio)
