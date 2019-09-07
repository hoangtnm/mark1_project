# This file presents an interface for interacting with the Playstation 4 Controller
# in Python. Simply plug your PS4 controller into your computer using USB and run this
# script!
#
# NOTE: I assume in this script that the only joystick plugged in is the PS4 controller.
#       if this is not the case, you will need to change the class accordingly.


import pygame
from threading import Thread


class PS4Controller:
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    def __init__(self):
        """Initialize the joystick components."""

        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        self.axis_data = False
        self.button_data = False
        self.hat_data = False

        # initialize the variable used to indicate if
        # the thread should be stopped
        self.stopped = False

    # Threading-method
    def start(self):
        # Start the thread to read signals from the controller
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next signal from the controller
            if not self.axis_data:
                self.axis_data = {0: 0.0, 1: 0.0, 2: 0.0,
                                  3: -1.0, 4: -1.0, 5: 0.0}    # default

            if not self.button_data:
                self.button_data = {}
                for i in range(self.controller.get_numbuttons()):
                    self.button_data[i] = False

            if not self.hat_data:
                self.hat_data = {}
                for i in range(self.controller.get_numhats()):
                    self.hat_data[i] = (0, 0)

            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    self.axis_data[event.axis] = round(event.value, 2)
                elif event.type == pygame.JOYBUTTONDOWN:
                    self.button_data[event.button] = True
                elif event.type == pygame.JOYBUTTONUP:
                    self.button_data[event.button] = False
                elif event.type == pygame.JOYHATMOTION:
                    self.hat_data[event.hat] = event.value

    def read(self):
        # return the signal most recently read
        return self.button_data, self.axis_data, self.hat_data

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def get_button_command(button_data, controller):
    """Get button number from a ps4 controller.

    Args:
        button_data: an array of length `controller.get_numbuttons()`
        controller: pygame.joystick.Joystick()
    Returns:
        is_command: a boolean value
        button_num: button number

    Button number map:
        0: SQUARE
        1: X
        2: CIRCLE
        3: TRIANGLE
        4: L1
        5: R1
        6: L2
        7: R2
        8: SHARE
        9: OPTIONSservoMin
        10: LEFT ANALOG PRESS
        11: RIGHT ANALOG PRESS
        12: PS4 ON BUTTON
        13: TOUCHPAD PRESS
    """

    is_command = False
    button_num = None
    total_buttons = controller.get_numbuttons()

    for num in range(total_buttons):
        if button_data[num]:
            is_command = True
            button_num = num
            break

    return is_command, button_num


if __name__ == "__main__":
    ps4 = PS4Controller()
    ps4.start()
    ps4.read()
