# deep_motion_planning
This repo contains various ROS packages dealing with end-to-end imitation learning for ground robot navigation using neural networks. 

More details about this work can be found in the papers about [imitation learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989182) and [imitation learning combined with reinforcement learning](https://arxiv.org/pdf/1805.07095.pdf).

## Usage

### With Move Base
In order to start the simulation, the navigation and load the static map, run:
```
roslaunch stage_worlds shapes.launch deep_motion_planning:=False
```

You can then start a mission by executing:
```
rosrun mission_control mission_control_node.py _mission_file:=$(rospack find mission_control)/missions/rooms.txt _deep_motion_planner:=False
```
The parameter <_mission_file> defines the path to the mission definition that you want to execute. 

### With Deep Motion Planner
To start the simulation in combination with the deep motion planner, run:
```
roslaunch stage_worlds shapes.launch deep_motion_planning:=True
```

You can then start a mission by executing:
```
rosrun mission_control mission_control_node.py _mission_file:=$(rospack find mission_control)/missions/rooms.txt _deep_motion_planner:=True
```
The parameter <_mission_file> defines the path to the mission definition that you want to execute. 

## Packages
### stage_worlds
The stage_worlds package contains the configuration files for the simulation, various world
definitions and their ROS launch files.

### mission_control
This package contains a mission control node which executes a user defined mission. Therefore,
a .txt file is parsed and a sequence of commands is generated. This sequence is then processed 
step-by-step. For more details on the definition of a mission, please refer to the README file
in the package directory.

### data_capture
The data_capture node subscribes to a set of topics and writes the time-synchronized messages
into a .csv file. Currently, the node records data from a laser scanner, the relative target 
pose for the navigation and the control commands that are send to the robot.

### deep_motion_planner
The package wraps a Tensorflow model for motion planning with a deep neural network. The node loads
a pretrained model and computes the control commands for the robot from raw sensor data.

### deep_learning_model
This is not a ROS package, but a independent project to train a Tensorflow model on data that is
captured with the *data_capture* package. The resulting trained model can then be loaded and executed
in the *deep_motion_planner* package
