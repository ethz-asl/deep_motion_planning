# deep_motion_planner

This package wraps a trained TensorFlow model as ROS node. It takes the a goal position, laser
scan data and the current robot pose to perform a motion planning task using a deep neural network. The action API of the `move_base` package is cloned to ensure its compatibility.

## Usage
```
roslaunch deep_motion_planner deep_motion_planner.launch
```
This will start the deep motion planner node and initialize it with a simple trained model.

The path to the model selected for planning is provided in the `deep_motion_planner.launch` file. There one also has to choose between the convolutional and the fully-connected model. The architecture needs to match with the pre-trained model weights.
