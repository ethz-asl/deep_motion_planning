# data_capture
The package recordes a set of ROS topics and writes the synchronized messages into a .csv file.

## Usage
```
rosrun data_capture data_capture_node.py
```
After receiving a message on the _start_ topic, the synchronized messages are captured into a
buffer. Currently, the node subscribes to a laser scan, a target pose and the cmd_vel topics.
It is important, that those message all contain a header as we need the timestamp of the messages
for synchronization.

The _stop_ topic ends the recording and writes the buffered data into a target_#.csv file where #
is a running sequence number. The _abort_ topic will clear the cache without writing the captured 
data into a file.

## Parameters
### storage_path (default: <data_capture folder>/data/)
The location where the written .csv files are put.
