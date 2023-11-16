import torch
from torchvision import models, transforms
from PIL import Image
import io
import json
import os

print("Using inference.py ....")
with open(s.path.join(model_dir, 'data_classes.txt'), 'rb') as f:
    CLASSES = [cls.strip() for cls in f.readlines()]

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
