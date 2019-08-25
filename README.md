# Formula Student Technion Driverless - Based on AirSim

This project is about training and implementing self-driving algorithm for Formula Student Driverless competitions. In such competitions, a formula race car, designed and built by students, is challenged to drive through previously unseen tracks that are marked by traffic cones.

We present a simulator for formula student car and the environment of a driverless competition. The simulator is based on AirSim.

<p align="center">
    <img src="images/technion_formula_car.png">
    The Technion Formula Student car. Actual car (left), simulated car (right)
</p>

The model of the Formula Student Technion car is provided by [Ryan Pourati](https://www.linkedin.com/in/ryanpo).

The environment scene is provided by [PolyPixel](https://www.polypixel3d.com/).

<p align="center">
    <img src="images/imitation_learning_real_example.gif">
    Driving in real-world using trained imitation learning model, based on AirSim data only
</p>

## Prerequisites

- Operating system: Windows 10 or Ubuntu
- GPU: Nvidia GTX 1080 or higher (recommended)
- Software: Unreal Engine 4.18 and Visual Studio - 2017 (see [upgrade instructions](https://github.com/Microsoft/AirSim/blob/master/docs/unreal_upgrade.md))
- AirSim: 1.2

## How to Use It

### 1. Choosing the Mode: Car, Multirotor or ComputerVision

By default AirSim will prompt you for choosing Car or Multirotor mode. You can use [SimMode](https://github.com/Microsoft/AirSim/blob/master/docs/settings.md#simmode) setting to specify the default vehicle to car (Formula Technion Student car).

### 2. Manual drive

If you have a steering wheel (Logitech G920) as shown below, you can manually control the car in the simulator. Also, you can use arrow keys to drive manually.

[More details](https://github.com/Microsoft/AirSim/blob/master/docs/steering_wheel_installation.md)

<p align="center">
    <img src="images/steering_wheel.gif">
</p>

### 3. Steering the car using imitation learning

Using imitation learning, we trained a deep learning model to steer a Formula Student car with an input of only one camera. Our code files for the training procedure are available [here](https://github.com/FSTDriverless/AirSim/tree/master/PythonClient/imitation_learning) and are based on [AirSim cookbook](https://github.com/Microsoft/AutonomousDrivingCookbook).

### 4. Gathering training data

We added a few [graphic features](https://github.com/Microsoft/AirSim/wiki/graphic_features) to ease the procedure of recording data.
You can change the positions of the cameras using [this tutorial](https://github.com/Microsoft/AirSim/wiki/cameras_positioning).

There are two ways you can generate training data from AirSim for deep learning. The easiest way is to simply press the record button on the lower right corner. This will start writing pose and images for each frame. The data logging code is pretty simple and you can modify it to your heart's desire.

<p align="center">
    <img src="images/recording_button_small.png">
</p>