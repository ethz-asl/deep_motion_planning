# deep_motion_planning
Using Deep Neural Networks for robot navigation purposes:

The repository contains various ROS packages that are used to generate the training data.

## Usage

In order to start the simulation, the navigation and load the static map, run:
```
roslaunch stage_worlds shapes.launch
```

You can then start a mission by executing:
```
rosrun mission_control mission_control_node.py _mission_file:=src/deep_motion_planning/mission_control/missions/rooms.txt
```
The parameter <_mission_file> defines the path to the mission definition that you want to execute. 
The example above assumes that you are in the top folder of your catkin workspace.

Finally, if you want to see a visualization of the system, run:
```
roslaunch stage_worlds rviz.launch
```

## Packages
### stage_worlds
The stage_worlds package contains the configuration files for the simulation, various world
definitions and their ROS launch files.

### mission_control
This package contains a mission control node which executes a user defined mission. Therefor,
a txt file is parsed and a sequence of commands is generated. This sequence is then processed 
step-by-step. For more details on the definition of a mission, please refer to the README file
in the package directory.

### data_capture
The data_capture node subscribes to a set of topics and writes the time-synchronized messages
into a .csv file. Currently, the node records data from a laser scanner, the relative target 
pose for the navigation and the control commands that are send to the robot.
