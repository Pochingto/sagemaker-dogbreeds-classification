import os
import argparse
import shutil
import random
import csv
import numpy as np

from PIL import Image
from pathlib import Path

IMAGE_STATS_CSV_NAME = "image_stats.csv"
IMAGE_CLASS_FOLDERS = ['Affenpinscher', 'Afghan_hound', 'Airedale_terrier', 'Akita','Alaskan_malamute','American_eskimo_dog','American_foxhound','American_staffordshire_terrier','American_water_spaniel',\
'Anatolian_shepherd_dog','Australian_cattle_dog','Australian_shepherd','Australian_terrier','Basenji','Basset_hound','Beagle','Bearded_collie','Beauceron','Bedlington_terrier',\
'Belgian_malinois','Belgian_sheepdog','Belgian_tervuren','Bernese_mountain_dog','Bichon_frise','Black_and_tan_coonhound','Black_russian_terrier','Bloodhound','Bluetick_coonhound','Border_collie',\
'Border_terrier','Borzoi','Boston_terrier','Bouvier_des_flandres','Boxer','Boykin_spaniel', 'Briard', 'Brittany', 'Brussels_griffon', 'Bull_terrier', 'Bulldog', 'Bullmastiff',\
'Cairn_terrier', 'Canaan_dog', 'Cane_corso', 'Cardigan_welsh_corgi', 'Cavalier_king_charles_spaniel', 'Chesapeake_bay_retriever', 'Chihuahua', 'Chinese_crested', 'Chinese_shar-pei', 'Chow_chow', \
'Clumber_spaniel', 'Cocker_spaniel', 'Collie', 'Curly-coated_retriever', 'Dachshund', 'Dalmatian', 'Dandie_dinmont_terrier', 'Doberman_pinscher', 'Dogue_de_bordeaux', 'English_cocker_spaniel',\
'English_setter', 'English_springer_spaniel', 'English_toy_spaniel', 'Entlebucher_mountain_dog', 'Field_spaniel', 'Finnish_spitz', 'Flat-coated_retriever', 'French_bulldog', 'German_pinscher', \
'German_shepherd_dog', 'German_shorthaired_pointer', 'German_wirehaired_pointer', 'Giant_schnauzer', 'Glen_of_imaal_terrier', 'Golden_retriever', 'Gordon_setter', 'Great_dane', \
'Great_pyrenees', 'Greater_swiss_mountain_dog', 'Greyhound', 'Havanese','Ibizan_hound','Icelandic_sheepdog','Irish_red_and_white_setter','Irish_setter','Irish_terrier','Irish_water_spaniel',\
'Irish_wolfhound','Italian_greyhound','Japanese_chin','Keeshond','Kerry_blue_terrier','Komondor','Kuvasz','Labrador_retriever','Lakeland_terrier','Leonberger','Lhasa_apso','Lowchen',\
'Maltese','Manchester_terrier','Mastiff','Miniature_schnauzer','Neapolitan_mastiff','Newfoundland','Norfolk_terrier','Norwegian_buhund','Norwegian_elkhound','Norwegian_lundehund',\
'Norwich_terrier','Nova_scotia_duck_tolling_retriever','Old_english_sheepdog','Otterhound','Papillon','Parson_russell_terrier','Pekingese','Pembroke_welsh_corgi','Petit_basset_griffon_vendeen',\
'Pharaoh_hound','Plott','Pointer','Pomeranian','Poodle','Portuguese_water_dog','Saint_bernard','Silky_terrier','Smooth_fox_terrier','Tibetan_mastiff','Welsh_springer_spaniel',\
'Wirehaired_pointing_griffon','Xoloitzcuintli','Yorkshire_terrier']

def split_data(data_dir, output_dir, train_ratio):
    print("Splitting data...")
    data_path = Path(data_dir)
    train_path = Path(output_dir) / 'train'
    test_path = Path(output_dir) / 'test'
    
    # Create train and test directories
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    for class_folder in IMAGE_CLASS_FOLDERS:
        print(f"Splitting class folder: {class_folder}")
        class_path = data_path / class_folder
        # check if is folder
        if not os.path.isdir(class_path):
            continue
            
        images = [file for file in os.listdir(class_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
    
def extract_rgb_features(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        img_array = np.array(img)

        if img_array.ndim == 3:
            red_channel = img_array[:, :, 0]
            green_channel = img_array[:, :, 1]
            blue_channel = img_array[:, :, 2]

            red_mean, red_std = np.mean(red_channel), np.std(red_channel)
            green_mean, green_std = np.mean(green_channel), np.std(green_channel)
            blue_mean, blue_std = np.mean(blue_channel), np.std(blue_channel)

            return [width, height, red_mean, red_std, green_mean, green_std, blue_mean, blue_std]
        else:
            return [width, height] + [None] * 6

def write_features_to_csv(output_csv, image_features):
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(image_features)
        
def extract_image_feature(train_data_dir, output_dir):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Initializing CSV...")
    output_csv = output_dir_path / IMAGE_STATS_CSV_NAME
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['width', 'height', 'red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std'])
    
    for class_name in os.listdir(train_data_dir):
        print(f"Extracting features in class {class_name} ...")
        class_path = os.path.join(train_data_dir, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                try: 
                    image_path = os.path.join(class_path, image_file)
                    features = extract_rgb_features(image_path)
                    write_features_to_csv(output_csv, features)
                except Exception as e: 
                    print(f"Processing image {image_path} exception: {e}")
    print("Feature extraction completed.")
                    
def preprocess_data(data_dir, output_dir, train_ratio):
    split_data(data_dir, output_dir, train_ratio)
    extract_image_feature(os.path.join(output_dir, "train"), os.path.join(output_dir, "train-baseline"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    args = parser.parse_args()

    preprocess_data(args.data_dir, args.output_dir, args.train_ratio)