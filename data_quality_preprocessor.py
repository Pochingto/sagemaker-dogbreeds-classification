import json
import numpy as np
import io
import base64

from PIL import Image

def extract_rgb_features(image_data):
    img = Image.open(io.BytesIO(image_data))
    width, height = img.size
    img_array = np.array(img)
    
    columns = ['width', 'height', 'red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std']
    if img_array.ndim == 3:
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        red_mean, red_std = np.mean(red_channel), np.std(red_channel)
        green_mean, green_std = np.mean(green_channel), np.std(green_channel)
        blue_mean, blue_std = np.mean(blue_channel), np.std(blue_channel)

        features = [width, height, red_mean, red_std, green_mean, green_std, blue_mean, blue_std]
    else:
        features = [width, height] + [-1.0] * 6
        
    return dict(zip(columns, features))

def preprocess_handler(inference_record):
    data = json.loads(inference_record.endpoint_input.data)
    image_data = data["image_data"]
    image_data = base64.b64decode(image_data)
    response = extract_rgb_features(image_data)
    return response