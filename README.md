# Formula Student Technion Driverless - Based on AirSim

This project is about training and implementing self-driving algorithm for `Formula Student Driverless competitions`. In such competitions, a formula race car, designed and built by students, is challenged to drive through previously unseen tracks that are marked by traffic cones.

We present a simulator for formula student car and the environment of a driverless competition. The simulator is based on AirSim.

<p align="center">
    <img src="images/technion_formula_car.png"><br>
    Figure 1. The Technion Formula Student car. Actual car (left), simulated car (right)
</p>

The model of the Formula Student Technion car is provided by [Ryan Pourati](https://www.linkedin.com/in/ryanpo).

The environment scene is provided by [PolyPixel](https://www.polypixel3d.com/).

<p align="center">
    <img src="images/imitation_learning_real_example.gif"><br>
    Figure 2. Driving in real-world using trained imitation learning model, based on AirSim data only
</p>

## 1. Prerequisites

- Operating system: Windows 10
- GPU: Nvidia GTX 1080 or higher (recommended)
- Software: Unreal Engine 4.18 and Visual Studio - 2017 (see [upgrade instructions](https://github.com/Microsoft/AirSim/blob/master/docs/unreal_upgrade.md))
- AirSim: 1.2

## 2. Build the AirSim Project

### 2.1. Install Unreal Engine

### 2.2. Build AirSim

1. You will need Visual Studio 2017 (**make sure** to install `VC++` and `Windows SDK 8.x`).
2. Start `x64 Native Tools Command Prompt for VS 2017`. Create a folder for the repo and run `git clone https://github.com/Microsoft/AirSim.git`.
3. Run `build.cmd` from the command line. This will create ready to use plugin bits in the `Unreal\Plugins` folder that can be dropped into any Unreal project.

### 2.3. Creating and Setting Up Unreal Environment

Finally, you will need an `Unreal project` that `hosts` the `environment` for your vehicles. Follow the list below to create an environment that simulates the FSD competitions.

1. Make sure `AirSim` is built and `Unreal 4.18` is installed as described above.

2. Open `UE editor` and choose `New Project`. Choose `Blank` with `no starter content`. Select your project's location, define it's name (`ProjectName` for example) and press `create project`.

<p align="center">
    <img src="images/unreal_new_project.png"><br>
    Figure 3. Create Unreal Project
</p>

3. After the project is loaded to the editor, from the `File` menu select `New C++ class`, leave default None on the type of class, click `Next`, leave default name MyClass, and click `Create Class`. We need to do this because Unreal requires at least one source file in project. It should trigger compile and `open up Visual Studio` solution `ProjectName.sln`.

4. Close and save `ProjectName.sln`. Also, close the UE editor.

5. Go to your folder for `AirSim` repo and copy `Unreal\Plugins` folder into your `ProjectName` folder. This way now your own Unreal project has AirSim plugin.

6. Download the environment assets of [FSD racecourse](https://drive.google.com/file/d/1FC1T8rZ5hVEDXwlECnPxmPitRCLlxGma/view?usp=sharing). Extract the zip into `ProjectName\Content` (see folders tree in the end of this doc).

7. Download the formula Technion [car assets](https://drive.google.com/file/d/1dV4deyLlmMwBwA2ljxbardbGdXHtKKSo/view?usp=sharing). Extract the zip into `ProjectName\Plugins\AirSim\Content\VehicleAdv\SUV` and select `replace` when asked for `SuvCarPawn.uasset` (the original file will be saved into a backup folder).

8. `Edit` the `ProjectName.uproject` so that it looks like this:

```json
{
	"FileVersion": 3,
	"EngineAssociation": "4.18",
	"Category": "Samples",
	"Description": "",
	"Modules": [
		{
			"Name": "ProjectName",
			"Type": "Runtime",
			"LoadingPhase": "Default",
			"AdditionalDependencies": [
				"AirSim"
			]
		}
	],
	"TargetPlatforms": [
		"MacNoEditor",
		"WindowsNoEditor"
	],
	"Plugins": [
		{
			"Name": "AirSim",
			"Enabled": true
		}
	]
}
```

9. Right click the `ProjectName.uproject` in Windows Explorer and select `Generate Visual Studio project files`. This step detects all plugins and source files in your Unreal project and generates .sln file for Visual Studio.

<p align="center">
    <img src="images/regen_sln.png"><br>
    Figure 4. Generate Visual Studio project files
</p>

## 3. How to Use It

### 3.1. Choosing the Mode: Car, Multirotor or ComputerVision

By default AirSim will prompt you for choosing Car or Multirotor mode. You can use [SimMode](https://github.com/Microsoft/AirSim/blob/master/docs/settings.md#simmode) setting to specify the default vehicle to car (Formula Technion Student car).

### 3.2. Manual drive

If you have a steering wheel (Logitech G920) as shown below, you can manually control the car in the simulator. Also, you can use arrow keys to drive manually.

[More details](https://github.com/Microsoft/AirSim/blob/master/docs/steering_wheel_installation.md)

<p align="center">
    <img src="images/steering_wheel.gif"><br>
    Figure 4. Manual drive
</p>

### 3.3. Steering the car using imitation learning

Using imitation learning, we trained a deep learning model to steer a Formula Student car with an input of only one camera. Our code files for the training procedure are available [here](https://github.com/FSTDriverless/AirSim/tree/master/PythonClient/imitation_learning) and are based on [AirSim cookbook](https://github.com/Microsoft/AutonomousDrivingCookbook).

### 3.4. Gathering training data

We added a few [graphic features](https://github.com/Microsoft/AirSim/wiki/graphic_features) to ease the procedure of recording data.
You can change the positions of the cameras using [this tutorial](https://github.com/Microsoft/AirSim/wiki/cameras_positioning).

There are two ways you can generate training data from AirSim for deep learning. The easiest way is to simply press the record button on the lower right corner. This will start writing pose and images for each frame. The data logging code is pretty simple and you can modify it to your heart's desire.

<p align="center">
    <img src="images/recording_button_small.png"><br>
    Figure n. Gathering training data
</p>

## References