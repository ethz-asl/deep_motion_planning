#!/usr/bin/env python2

import rospy
from roslib.packages import get_pkg_dir 

from data_capture import DataCapture

default_storage_path = get_pkg_dir('data_capture') + '/data/'

def main():

    rospy.init_node('data_capture_node')

    storage_path = rospy.get_param('storage_path',default_storage_path)
    DataCapture(storage_path)

    rospy.spin()

if __name__ == "__main__":
    main()
