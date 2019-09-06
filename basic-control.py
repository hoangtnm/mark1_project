import airsim
import numpy as np
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

    # restore to original state
    client.reset()

    client.enableApiControl(False)
