import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
import logging
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped


def min_pool_laser(laser_data, num_chunks):
  """
  Subsample laser measurements (take minimum of each chunk).
  Inputs:
    laser_data: raw laser measurements of size [batch_size, number measurements]
    num_chunks: number of chunks in which the laser data should be summarized
  """
  num_laser = laser_data.shape[1]

  values_per_chunk = int(num_laser / num_chunks)

  laser_min_chunks = np.zeros([laser_data.shape[0], num_chunks])

  for ii in range(num_chunks):
    laser_min_chunks[:, ii] = np.min(laser_data[:, ii*values_per_chunk:(ii+1)*values_per_chunk], axis=1)

  return laser_min_chunks


def crop_laser(laser_data, max_range=30.0):
  return np.minimum(laser_data, max_range)


def normalize_laser(laser_data, max_range=30):
  return laser_data / max_range

def invert_laser(laser_data):
  return 1 - laser_data

def transform_laser(laser_data, num_chunks=36, max_range=30.0):
  laser = min_pool_laser(laser_data, num_chunks)
  laser = crop_laser(laser, max_range)
  laser = normalize_laser(laser, max_range)
  laser = invert_laser(laser)
  return 2*laser - 1

def transform_target_distance(target_distance, norm_range=30.0):
  tmp = np.minimum(target_distance, norm_range)
  tmp = 1 - tmp / norm_range
  return 2 * tmp - 1

def transform_target_angle(target_angle, norm_angle=np.pi):
  return target_angle / norm_angle

def get_target_in_robot_frame(robot_pose_global_frame, target_pose_global_frame):
  """
  Get the target position in the robot frame in cylindrical coordinates.
  Inputs: robot and target pose in global frame (numpy arrays with [x, y, yaw])
  Output: [distance to goal, angle in local robot frame, heading of the target in the local robot frame]
  """
  target = PoseStamped()
  goal_position_difference = [target_pose_global_frame[0] - robot_pose_global_frame[0],
                              target_pose_global_frame[1] - robot_pose_global_frame[1]]

  # Get the quaternion from the current goal
  q = tf.transformations.quaternion_from_euler(0., 0., target_pose_global_frame[2])

  p = tf.transformations.quaternion_from_euler(0., 0., robot_pose_global_frame[2])

  # Rotate the relative goal position into the base frame
  goal_position_base_frame = tf.transformations.quaternion_multiply(tf.transformations.quaternion_inverse(p),
                                                                    tf.transformations.quaternion_multiply([goal_position_difference[0],
                                                                                                            goal_position_difference[1], 0, 0], p))

  # Compute the difference to the goal orientation
  orientation_to_target = tf.transformations.quaternion_multiply(q, tf.transformations.quaternion_inverse(p))
  target.pose.orientation.x = orientation_to_target[0]
  target.pose.orientation.y = orientation_to_target[1]
  target.pose.orientation.z = orientation_to_target[2]
  target.pose.orientation.w = orientation_to_target[3]

  target.pose.position.x = goal_position_base_frame[0]
  target.pose.position.y = goal_position_base_frame[1]

  return target

def get_yaw_from_quat(orientation):
  p = [orientation.x, orientation.y, orientation.z, orientation.w]
  return tf.transformations.euler_from_quaternion(p)[2]


#### Some minor functions to compare quaternion transformation
def get_distance(pos1, pos2):
    return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)

def get_angle(pos1, pos2):
  return np.arctan2(pos2.y - pos1.y, pos2.x - pos1.x)

def get_relative_angle_to_goal(position, orientation, goal_position):
  angle = get_angle(position, goal_position) - tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]
  if(angle/np.pi < -1):
    return angle + 2*np.pi
  elif(angle/np.pi > 1):
    return angle - 2*np.pi
  return angle


def transform_laser_tai(laser_data):
  max_range = 10.0
  number_samples = 10.0
  laser_data = np.minimum(laser_data, max_range)
  laser_data = laser_data / max_range
  total_laser_samples = laser_data.shape[1]
  network_laser_inputs = 10
  laser_sensor_offset = 0.0
  laser_slice = int((2*total_laser_samples)/(3*network_laser_inputs))
  laser_slice_offset = int(total_laser_samples/6)
  laser_idxs = np.array(range(total_laser_samples))
  laser_idxs_sampled = laser_idxs[laser_slice_offset+int(laser_slice/2):total_laser_samples - laser_slice_offset:laser_slice]
#   print("laser idxs {}".format(laser_idxs_sampled))
  laser_data_sampled = laser_data[:, laser_idxs_sampled] - laser_sensor_offset
  return laser_data_sampled


def transform_laser_mix(laser_data):
  max_range = 10.0
  num_chunks = 10
  # remove values on left and right to get 180 deg FOV
  number_cut_laser_values =180
  laser_data = laser_data[:, number_cut_laser_values:-number_cut_laser_values]
  laser = transform_laser(laser_data, num_chunks=num_chunks, max_range=max_range)
  return laser


