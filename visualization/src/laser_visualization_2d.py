#!/usr/bin/env python
import rospy

import cv2
import numpy as np
import math
import tf

# Messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

class LaserVisualization2d(): 
  """
  Visualize laser data of 2D laser range finders
  """
  def __init__(self):
    self.laser_sub = rospy.Subscriber("/base_scan", LaserScan, self.laser_scan_callback)
    self.relative_target_sub = rospy.Subscriber("/relative_target", PoseStamped, self.relative_target_callback)
    self.relative_target = None
    self.relative_target_update_time = None

  def laser_scan_callback(self, data):
    frame = np.ones((500, 500,3), np.uint8)*255
    
    # Laser scans
    angle = data.angle_min
    for r in data.ranges:
      x = math.trunc((r * 30) * math.cos(angle + (-90*math.pi/180)))
      y = math.trunc((r * 30) * math.sin(angle + (-90*math.pi/180)))
      cv2.line(frame, (250, 250), (-x+250,y+250), (0,0,255), 2)
      angle= angle + data.angle_increment
      
    
    # Target
    if self.do_plot_target():
      target_pos = (250+int(self.relative_target[1]*30), 250-int(self.relative_target[0]*30))
      length = 15.0
      color_target = (0, 255, 0)
      x = math.trunc(length * math.cos(self.relative_target[2] - math.pi/2))
      y = math.trunc(length * math.sin(self.relative_target[2] - math.pi/2))
      cv2.circle(img=frame, center=target_pos, radius=4, thickness=2, color=color_target)
      cv2.line(frame, target_pos, (-x+target_pos[0],y+target_pos[1]), color_target, 2)
    
    # Robot
    cv2.circle(img=frame, center=(250, 250), radius=4, thickness=7, color=(0, 0, 0))
    cv2.line(frame, (250, 250), (250,244), (255,255,255), 2)
    cv2.imshow('laser',frame)
    cv2.waitKey(1)
    
  def relative_target_callback(self, data):
    x = data.pose.position.x
    y = data.pose.position.y
    phi = tf.transformations.euler_from_quaternion([data.pose.orientation.x,
                                                   data.pose.orientation.y,
                                                   data.pose.orientation.z,
                                                   data.pose.orientation.w])[2]
    self.relative_target = [x, y, phi]
    self.relative_target_update_time = rospy.Time.now()
    
  def do_plot_target(self):
    if self.relative_target_update_time == None or self.relative_target == None: return False
    if (rospy.Time.now() - self.relative_target_update_time).to_sec() > 2: return False
    return True

  def __enter__(self):
    return self
      
  def __exit__(self, exc_type, exc_value, traceback):
    return self
