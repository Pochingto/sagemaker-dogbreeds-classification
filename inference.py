import torch
from torchvision import models, transforms
from PIL import Image
import io
import json
import os

print("Using inference.py ....")
CLASSES = ['Affenpinscher', 'Afghan_hound', 'Airedale_terrier', 'Akita', 'Alaskan_malamute', 'American_eskimo_dog', 'American_foxhound', 'American_staffordshire_terrier', 'American_water_spaniel', 'Anatolian_shepherd_dog', 'Australian_cattle_dog', 'Australian_shepherd', 'Australian_terrier', 'Basenji', 'Basset_hound', 'Beagle', 'Bearded_collie', 'Beauceron', 'Bedlington_terrier', 'Belgian_malinois', 'Belgian_sheepdog', 'Belgian_tervuren', 'Bernese_mountain_dog', 'Bichon_frise', 'Black_and_tan_coonhound', 'Black_russian_terrier', 'Bloodhound', 'Bluetick_coonhound', 'Border_collie', 'Border_terrier', 'Borzoi', 'Boston_terrier', 'Bouvier_des_flandres', 'Boxer', 'Boykin_spaniel', 'Briard', 'Brittany', 'Brussels_griffon', 'Bull_terrier', 'Bulldog', 'Bullmastiff', 'Cairn_terrier', 'Canaan_dog', 'Cane_corso', 'Cardigan_welsh_corgi', 'Cavalier_king_charles_spaniel', 'Chesapeake_bay_retriever', 'Chihuahua', 'Chinese_crested', 'Chinese_shar-pei', 'Chow_chow', 'Clumber_spaniel', 'Cocker_spaniel', 'Collie', 'Curly-coated_retriever', 'Dachshund', 'Dalmatian', 'Dandie_dinmont_terrier', 'Doberman_pinscher', 'Dogue_de_bordeaux', 'English_cocker_spaniel', 'English_setter', 'English_springer_spaniel', 'English_toy_spaniel', 'Entlebucher_mountain_dog', 'Field_spaniel', 'Finnish_spitz', 'Flat-coated_retriever', 'French_bulldog', 'German_pinscher', 'German_shepherd_dog', 'German_shorthaired_pointer', 'German_wirehaired_pointer', 'Giant_schnauzer', 'Glen_of_imaal_terrier', 'Golden_retriever', 'Gordon_setter', 'Great_dane', 'Great_pyrenees', 'Greater_swiss_mountain_dog', 'Greyhound', 'Havanese', 'Ibizan_hound', 'Icelandic_sheepdog', 'Irish_red_and_white_setter', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_chin', 'Keeshond', 'Kerry_blue_terrier', 'Komondor', 'Kuvasz', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberger', 'Lhasa_apso', 'Lowchen', 'Maltese', 'Manchester_terrier', 'Mastiff', 'Miniature_schnauzer', 'Neapolitan_mastiff', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_buhund', 'Norwegian_elkhound', 'Norwegian_lundehund', 'Norwich_terrier', 'Nova_scotia_duck_tolling_retriever', 'Old_english_sheepdog', 'Otterhound', 'Papillon', 'Parson_russell_terrier', 'Pekingese', 'Pembroke_welsh_corgi', 'Petit_basset_griffon_vendeen', 'Pharaoh_hound', 'Plott', 'Pointer', 'Pomeranian', 'Poodle', 'Portuguese_water_dog', 'Saint_bernard', 'Silky_terrier', 'Smooth_fox_terrier', 'Tibetan_mastiff', 'Welsh_springer_spaniel', 'Wirehaired_pointing_griffon', 'Xoloitzcuintli', 'Yorkshire_terrier']

# Define the model_fn which loads the model
def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    """
    print("Loading model.")
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    num_classes = 133
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the saved model parameters
    model_path = os.path.join(model_dir, 'model.pt')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    
    model.eval()
    return model

# Define the input_fn which handles the preprocessing of the input data
def input_fn(request_body, request_content_type):
    """
    Preprocess the incoming image data for prediction.
    """
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        image = image.unsqueeze(0)

        return image

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# Define the predict_fn which performs the prediction
def predict_fn(input_data, model):
    """
    Perform prediction using the model and the processed input data.
    """
    print("Predicting class labels for the input data...")
    
    model.eval()
    
    with torch.no_grad():
        output = model(input_data)
    
    return output

# Define the output_fn which formats the prediction output
def output_fn(prediction_output, accept):
    """
    Format the prediction output to the desired format.
    """
    print('Formatting prediction output...')
    
    if "application/json" in accept:
        # Convert the prediction output to JSON
        prediction = prediction_output.argmax(1).tolist()
        print(prediction)
        try: 
            response = json.dumps(CLASSES[prediction[0]])
            return response
        except Exception as e: 
            print("Prediction: ", prediction)
            print(e)
    else:
        raise Exception(f"Requested unsupported ContentType in Accept: {accept}")
