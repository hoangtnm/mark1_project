import os
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing import image
import tensorflow.keras.backend as K
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms
import airsim
import numpy as np
import time
import sys
from keras.models import load_model
from PIL import Image

from train_model import NeuralNet


# Trained model path
MODEL_PATH = './models/example_model.h5'

model = load_model(MODEL_PATH)

# Connect to AirSim
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

# Start driving
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0
client.setCarControls(car_controls)

# Initialize image buffer
image_buf = np.zeros((1, 66, 200, 3))


def get_image():
    """Gets image from AirSim client."""
    image_response = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgb = img1d.reshape(image_response.height, image_response.width, 3)
    return image_rgb


while True:
    # Update throttle value according to steering angle
    if abs(car_controls.steering) <= 1.0:
        car_controls.throttle = 0.8-(0.4*abs(car_controls.steering))
    else:
        car_controls.throttle = 0.4

    image_buf[0] = get_image()
    image_buf[0] /= 255  # Normalization

    start_time = time.time()

    # Prediction
    model_output = model.predict([image_buf])

    end_time = time.time()
    received_output = model_output[0][0]

    # Rescale prediction to [-1,1] and factor by 0.82 for drive smoothness
    car_controls.steering = round(
        (0.82*(float((model_output[0][0]*2.0)-1))), 2)

    # Print progress
    print('Sending steering = {0}, throttle = {1}, prediction time = {2}'.format(
        received_output, car_controls.throttle, str(end_time-start_time)))

    # Update next car state
    client.setCarControls(car_controls)

    # Wait a bit between iterations
    time.sleep(0.05)


client.enableApiControl(False)

if __name__ == "__main__":
    MAX_SPEED = 8

    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    data_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet()
    model.load_state_dict(torch.load('models/checkpoint.pth'))
    model.to(device)

    try:
        # reset the car state
        car_controls.steering = 0
        car_controls.throttle = 0
        car_controls.brake = 0
        client.setCarControls(car_controls)

        while True:
            # Update throttle value according to steering angle
            if abs(car_controls.steering) <= 1.0:
                car_controls.throttle = 0.8-(0.4*abs(car_controls.steering))
            else:
                car_controls.throttle = 0.4

            image = get_image()
            image = Image.fromarray(image)
            image = data_transforms(image).unsqueeze_(0)
            image.to(device)

            start_time = time.time()

            # Prediction
            angle = model(image).item()

            end_time = time.time()

            # Scales prediction to [-1,1]
            # and factor by 0.82 for drive smoothness
            angle = round(0.82*(angle * 2 - 1), 2)

            print(f'Sending steering = {received_output}, throttle = {car_controls.throttle}, \
                    prediction time = {end_time-start_time}.')

            # Updates next car-state
            car_controls.steering = angle
            client.setCarControls(car_controls)

            # Wait a bit between iterations
            time.sleep(0.05)

    except KeyboardInterrupt:
        # restore to original state
        client.reset()

        client.enableApiControl(False)
