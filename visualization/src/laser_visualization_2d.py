#!/usr/bin/env python
import rospy

import cv2
import numpy as np
import math

from sensor_msgs.msg import LaserScan

class LaserVisualization2d(): 
  """
  Visualize laser data of 2D laser range finders
  """
  def __init__(self):
    self.laser_sub = rospy.Subscriber("/base_scan", LaserScan, self.laser_scan_callback)

  def laser_scan_callback(self, data):
    frame = np.zeros((500, 500,3), np.uint8)
    angle = data.angle_min
    for r in data.ranges:
      x = math.trunc((r * 30) * math.cos(angle + (-90*3.1416/180)))
      y = math.trunc((r * 30) * math.sin(angle + (-90*3.1416/180)))
      cv2.line(frame, (250, 250), (x+250,y+250), (0,0,255), 2)
      angle= angle + data.angle_increment

    cv2.circle(frame, (250, 250), 2, (255, 0, 0))
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

  def __enter__(self):
    return self
      
  def __exit__(self, exc_type, exc_value, traceback):
    return self
