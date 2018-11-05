# deep_motion_planning
Using Deep Neural Networks for map-less end-to-end robot navigation purposes:

The repository contains various packages to record the training data, train the end-to-end model and deploy it on a robot -- in simulation or in our case a real Turtlebot.

This code is related to the following publications:
- [From Perception to Decision: A Data-driven Approach to End-to-end
Motion Planning for Autonomous Ground Robots](https://arxiv.org/pdf/1609.07910.pdf)
- [Reinforced Imitation: Sample Efficient Deep
Reinforcement Learning for Map-less Navigation by
Leveraging Prior Demonstrations](https://arxiv.org/pdf/1805.07095.pdf)

## Usage
The robot can be controlled basd on the ROS `move_base` module or using the learned `deep_motion_planner`. For training data recording, the `move_base` planner has to be used. 

### Navigating with move_base
In order to start the simulation, the navigation and load the static map, run:
```
roslaunch stage_worlds shapes.launch deep_motion_planning:=False
```

`shapes.launch` launches the environment and can be replaced with the desired map. 

A visualization of the system can be shown using RViz. There one can also select desired target positions by using the `2D Nav Goal` button.

An autonomous waypoint navigation mission (where no waypoints have to be selected manually) can be started with
```
rosrun mission_control mission_control_node.py _mission_file:=$(rospack find mission_control)/missions/shapes.txt _deep_motion_planner:=False
```
The parameter `_mission_file` defines the path to the mission definition that should be executed. 


### Navigation with deep_motion_planner
To start the simulation in combination with the deep motion planner, run:
```
roslaunch stage_worlds shapes.launch deep_motion_planning:=True
```

Here, the waypoint navigation is started with
```
rosrun mission_control mission_control_node.py _mission_file:=$(rospack find mission_control)/missions/shapes.txt _deep_motion_planner:=True
```


## Packages
### stage_worlds
The stage_worlds package contains the configuration files for the simulation, the navigation modules, various world
definitions and their according launch files.

### mission_control
This package contains a mission control node which executes a user waypoint navigation mission. Therefore,
a `.txt` file is parsed and a sequence of commands is generated. This sequence is then processed 
step-by-step. For more details on the definition of a mission, please refer to the README file
in the package directory.

### data_capture
The data_capture node subscribes to a set of topics and writes the time-synchronized messages
into a `.csv` file. Currently, the node records data from a laser scanner, the relative target 
pose for the navigation and the control commands that are send to the robot.

### deep_motion_planner
The package wraps a TensorFlow model for motion planning with a deep neural network. The node loads
a pretrained model and computes the control commands for the robot from raw sensor data.

### deep_learning_model
This is not a ROS package, but a independent project to train a TensorFlow model on data that is
captured with the *data_capture* package. The resulting trained model can then be loaded and executed
in the *deep_motion_planner* package. 
