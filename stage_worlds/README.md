# stage_worlds
This package is a wrapper around the stage_ros package which contains the simulator stage and a
ROS node that makes the simulation accessible as ROS topic.
This package contains various configuration files for stage that define the Europa robot, various 
worlds and ROS specific configuration and launch files.

## Usage
```
roslaunch stage_worlds shapes.launch
```
This start the simulation, the ROS navigation and loads the static map server.

## Worlds
A world is defined as grayscale image (see the ./worlds/bitmaps folder for examples)

These are the worlds that are defined:
### empty.world
![empty](./worlds/bitmaps/empty.png)
### simple.world
![simple](./worlds/bitmaps/simple.png)
### shapes.world
![shapes](./worlds/bitmaps/shapes.png)
### rooms.world
![rooms](./worlds/bitmaps/rooms.png)
### boxes.world
![boxes](./worlds/bitmaps/boxes.png)
