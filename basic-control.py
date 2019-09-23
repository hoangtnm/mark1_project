import setup_path
import cv2
import airsim
import time
import numpy as np
from PIL import Image
from controller import PS4Controller


MAX_SPEED = 1
MAX_THETA = 1


if __name__ == '__main__':
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    controller = PS4Controller()
    controller.start()

    """PS4 Controller's axes map

                 Axis 1                  Axis 4

                  -1                      -1
                   ^                       ^
                   |                       |
    Axis 0  -1 <---0---> 1   Axis 3 -1 <---0---> 1
                   |                       |
                   v                       v
                   1                       1
    """

    while True:
        _, axis_data, _ = controller.read()
        raw_speed = axis_data[1]
        raw_theta = axis_data[3]
        speed = np.interp(-raw_speed, (-1, 1), (-MAX_SPEED, MAX_SPEED))
        theta = np.interp(raw_theta, (-1, 1), (-MAX_THETA, MAX_THETA))

        car_controls.throttle = speed
        car_controls.steering = theta
        client.setCarControls(car_controls)

        car_state = client.getCarState()
        print(f'Speed {car_state.speed}, Gear {car_state.gear}')

        # Getting images
        responses = client.simGetImages([
            # png format
            airsim.ImageRequest("0", airsim.ImageType.Scene),
            # uncompressed RGB array bytes
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False),
            # floating point uncompressed image
            airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)
        ])

        response = responses[1]
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # reshape array to 3 channel image array H X W X 3
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        cv2.imshow('image', img_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # restore to original state
    client.reset()

    client.enableApiControl(False)
