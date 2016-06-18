# deep_motion_planner

This package wraps a trained Tensorflow model as ROS node. It takes the a goal position, laser
scan data and the current robot pose to perform a motion planning using a deep neural network. We
tried to imitate the action API of the move pase package to ensure its compatibility.

## Usage
```
roslaunch deep_motion_planner deep_motion_planner.launch
```
This will start the deep motion planner node and initialize it with a simple trained model.
