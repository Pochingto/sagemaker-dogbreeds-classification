import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import json
import os
import base64

print("Using inference.py ....")

# Define the model_fn which loads the model
def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    """
    print("Loading model.")
    with open(os.path.join(model_dir, 'data_classes.txt'), 'r') as f:
        classes = [cls.strip() for cls in f.readlines()]
        
    print(f"CLASSES: {classes}")
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    num_classes = len(classes)
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the saved model parameters
    model_path = os.path.join(model_dir, 'model.pt')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    
    model.eval()
    print("model_fn completed.")
    return {
        "model": model, 
        "classes": classes
    }

# Define the input_fn which handles the preprocessing of the input data
def input_fn(request_body, request_content_type):
    """
    Preprocess the incoming image data for prediction.
    """
    print("request_content_type: ", request_content_type)
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body))
    elif request_content_type == "application/json": 
        image_data = json.loads(request_body)["image_data"]
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    print("Image loaded.")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image = image.unsqueeze(0)
    print("input_fn complete")
    return image

# Define the predict_fn which performs the prediction
def predict_fn(input_data, model):
    """
    Perform prediction using the model and the processed input data.
    """
    print("Predicting class labels for the input data...")
    classes = model["classes"]
    model = model["model"]
    model.eval()
    
    with torch.no_grad():
        output = model(input_data)
    print("predict_fn completed.")
    return {
        "prediction_output": output,
        "classes": classes
    }

# Define the output_fn which formats the prediction output
def output_fn(prediction_output, accept):
    """
    Format the prediction output to the desired format.
    """
    print('Formatting prediction output...')
    classes = prediction_output["classes"]
    prediction_output = prediction_output["prediction_output"]
    
    if "application/json" in accept:
        # Convert the prediction output to JSON
        probs = F.softmax(prediction_output, 1)
        conf, pred = torch.max(probs, 1)
        conf = conf.tolist()[0]
        pred = pred.tolist()[0]
        try: 
            response = json.dumps({
                "prediction": classes[pred],
                "confidence": conf
            })
            
            return response
        except Exception as e: 
            print("Prediction: ", prediction)
            print(e)
    else:
        raise Exception(f"Requested unsupported ContentType in Accept: {accept}")
