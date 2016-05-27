
import rospy
import message_filters
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from geometry_msgs.msg import TwistStamped, PoseStamped

import tf

import csv
import os
from datetime import datetime

class DataCapture():
    """docstring for DataCapture"""
    def __init__(self, storage_path):

        # Create a new folder for each experiment
        # Name it after the current time
        date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
        self.storage_path = os.path.join(storage_path, date_str)

        rospy.loginfo(self.storage_path)

        self.enable_capture = False
        self.data_buffer = list()
        self.target_count = 1
        self.first_file = True

        # ROS topics
        rospy.Subscriber('/start', Empty, self.start_callback)
        rospy.Subscriber('/stop', Empty, self.stop_callback)
        rospy.Subscriber('/abort', Empty, self.abort_callback)

        # Synchronized messages
        scan_sub = message_filters.Subscriber('scan', LaserScan)
        cmd_sub = message_filters.Subscriber('cmd_vel', TwistStamped)
        target_sub = message_filters.Subscriber('relative_target', PoseStamped)

        self.synchonizer = message_filters.TimeSynchronizer([scan_sub, cmd_sub, target_sub], 10)
        self.synchonizer.registerCallback(self.sync_callback)

    def start_callback(self, data):
        if not self.enable_capture:
            rospy.loginfo('Start data capture')
            self.enable_capture = True

    def stop_callback(self, data):
        if self.enable_capture:
            rospy.loginfo('Stop data capture')
            self.enable_capture = False

            self.__write_data_to_file__()

    def abort_callback(self, data):
        if self.enable_capture:
            rospy.loginfo('Abort and clear buffered data')
            self.data_buffer = list()
            self.enable_capture = False

    def sync_callback(self, scan, cmd, target):
        if self.enable_capture:
            # concatenate the data and add it to the buffer
            orientation = [target.pose.orientation.x, target.pose.orientation.y,
                    target.pose.orientation.z, target.pose.orientation.w]
            yaw = tf.transformations.euler_from_quaternion(orientation)[2]
            new_row = [cmd.header.stamp.to_nsec(), cmd.twist.linear.x, cmd.twist.angular.z] + \
                    list(scan.ranges) + [target.pose.position.x, target.pose.position.y, yaw]
            self.data_buffer.append(new_row)

    def __write_data_to_file__(self):
        rospy.loginfo('Write data to file: {} items'.format(len(self.data_buffer)))

        # Prevent creation of empty files
        if len(self.data_buffer) == 0:
            rospy.loginfo('Received no messages: No csv file is written')
            return

        # Create the storage folder when writing the first file
        if self.first_file:
            os.mkdir(self.storage_path)
            self.first_file = False

        # Create the first line of the csv file with column names
        column_line = ['stamp','linear_x','angular_z'] + \
                ['laser_' + str(i) for i in range(len(self.data_buffer[0]) - 3)] + ['target_x',
                        'target_y', 'target_yaw']

        # write the data into a csv file and reset the buffer
        with open(os.path.join(self.storage_path,('target_' + str(self.target_count) + '.csv')), \
            'wb') as output_file:

            writer = csv.writer(output_file, delimiter=',')
            writer.writerow(column_line)
            for l in self.data_buffer:
                writer.writerow(l)

        self.data_buffer = list()
        self.target_count += 1

