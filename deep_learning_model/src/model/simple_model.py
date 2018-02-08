"""
This file contains methods that describe a Tensorflow model.
It can be used as template for a completely new model and is imported
in the training script
"""
import logging

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, conv2d, xavier_initializer_conv2d, xavier_initializer, l1_regularizer
import tensorflow.contrib as contrib

# Give the model a descriptive name
NAME = 'shapes_wo_heading'

# The size of the input layer
N_RANGE_FINDINGS = 36
N_TARGET = 2
INPUT_SIZE = N_RANGE_FINDINGS + N_TARGET

# The size of the output layer
CMD_SIZE = 2

# Laser preprocessing
N_LASER_MEASUREMENTS = 1080
MEASUREMENTS_PER_SLICE = N_LASER_MEASUREMENTS / N_RANGE_FINDINGS



# Helper functions
def exp_transform(distance, decay_factor = 0.2):
  return np.exp(-decay_factor*distance)


def get_laser_data_states(laser_data, n_laser_per_slice):
  """
  Takes laser measurements and downsample them by taking the minimum within each of the slice sectors.
  inputs:
    laser_data: input laser measurements
    n_laser_per_slice: number of laser measurements
  """
  laser_data_states = 2*exp_transform(np.array( [min(laser_data.ranges[current: current+laser_slice]) for current in xrange(0, len(laser_data.ranges), laser_slice)])) - 1
  return list(laser_data_states)

def get_pose_data_states(pose_data, goal):
  """
  Takes the pose data of the robot as input, along with the pose of goal (both are of type Pose()) and returns the two states to be fed into the network
  """
  position_data_state = 2*doExponentialTransform(getDistance(pose_data.position, goal.position)) - 1
  orientation_to_goal_data_state = getRelativeAngleToGoal(pose_data.position, pose_data.orientation, goal.position)/np.pi
  return [position_data_state] + [orientation_to_goal_data_state]


def learning_rate(initial):
  """
  Define learning rate for your model
  """
  # We use an exponential decy for the model
  global_step = tf.Variable(0, name='global_step', trainable=False)
  learning_rate = tf.train.exponential_decay(initial, global_step,
      250000, 0.85, staircase=True)
  return global_step, learning_rate

def __get_variable__(index, input_size, output_size):
    return (tf.get_variable('weights_hidden{}'.format(index), shape=[input_size, output_size],
            initializer=tf.contrib.layers.xavier_initializer()),
            tf.get_variable('biases_hidden{}'.format(index), [output_size]))

def _conv_layer(input, weights, biases, conv_stride_length=1, padding="SAME", name="conv", summary=False):
  """
  Standard 2D convolutional layer
  """
  conv = tf.nn.conv2d(input, filter=weights, strides=[1, conv_stride_length, conv_stride_length, 1], padding=padding, name=name)
  activations = tf.nn.relu(conv + biases)
  if summary:
    tf.summary.histogram(name, activations)
  return activations


def _fc_layer(input, weights, biases, use_activation=False, name="fc", summary=False):
  """
  Fully connected layer with given weights and biases.
  Activation and summary can be activated with the arguments.
  """
  affine_result = tf.matmul(input, weights) + biases
  if use_activation:
    activations = tf.nn.sigmoid(affine_result)
  else:
    activations = affine_result
  if summary:
    tf.summary.histogram(name + "_activations", activations)
  return activations

def _get_weight_variable(shape, name, regularizer=None,
                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                         summary=True, trainable=True):
  """
  Get weight variable with specific initializer.
  """
  var = tf.get_variable(name=name, shape=shape, initializer=initializer,
                        regularizer=tf.contrib.layers.l2_regularizer(0.01),
                        trainable=trainable)
  if summary:
    tf.summary.histogram(name, var)
  return var

def _get_bias_variable(shape, name, regularizer=None,
                       initializer=tf.constant_initializer(0.1),
                       summary=True, trainable=True):
  """
  Get bias variable with specific initializer.
  """
  var = tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=trainable)
  if summary:
    tf.summary.histogram(name, var)
  return var


def inference(data, keep_prob, sample_size, training=True, reuse=False, regularization_weight=0.001, output_name='prediction'):
  """
  Define the deep neural network used for inference
  """
  # Slice input data from data tensor
#   laser = tf.slice(data, [0, 0], [sample_size, N_RANGE_FINDINGS])
#   goal = tf.slice(data, [0, N_RANGE_FINDINGS], [sample_size, N_TARGET])
#
#   state = tf.concat([laser, goal], axis=1, name='state')

  n_hidden1 = 1000
  n_hidden2 = 300
  n_hidden3 = 100

  with tf.variable_scope('Weights', reuse=tf.AUTO_REUSE):
     weights = {'h1' : _get_weight_variable(shape=[INPUT_SIZE, n_hidden1], name='h1', regularizer=None,
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                            summary=True, trainable=training),
                'h2' : _get_weight_variable(shape=[n_hidden1, n_hidden2], name='h2', regularizer=None,
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                            summary=True, trainable=training),
                'h3' : _get_weight_variable(shape=[n_hidden2, n_hidden3], name='h3', regularizer=None,
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                            summary=True, trainable=training),
                'out' : _get_weight_variable(shape=[n_hidden3, CMD_SIZE], name='out', regularizer=None,
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                            summary=True, trainable=training)}
  with tf.variable_scope('Biases', reuse=tf.AUTO_REUSE):
    biases = {'b1' : _get_bias_variable(shape=[n_hidden1], name='b1', trainable=training),
              'b2' : _get_bias_variable(shape=[n_hidden2], name='b2', trainable=training),
              'b3' : _get_bias_variable(shape=[n_hidden3], name='b3', trainable=training),
              'out' : _get_bias_variable(shape=[CMD_SIZE], name='out', trainable=training)}


  hidden_layer1 = tf.nn.tanh(tf.add(tf.matmul(data, weights['h1']), biases['b1']))
  hidden_layer2 = tf.nn.tanh(tf.add(tf.matmul(hidden_layer1, weights['h2']), biases['b2']))
  hidden_layer3 = tf.nn.tanh(tf.add(tf.matmul(hidden_layer2, weights['h3']), biases['b3']))
  prediction = tf.nn.tanh(tf.add(tf.matmul(hidden_layer3, weights['out']), biases['out']))

  return prediction

def loss(prediction, cmd):
  """
  Define the loss used during the training steps
  """
  loss_split = tf.reduce_mean(tf.abs(prediction - cmd), 0)
  loss = tf.reduce_mean(tf.abs(prediction - cmd))

  return loss, loss_split

def training(loss, loss_split, learning_rate, global_step):
  """
  Perform a single training step optimization
  """
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)

  return train_op

def evaluation(prediction, cmd):
  """
  Define how to evaluate the model
  """
  loss_split = tf.reduce_mean(tf.abs(prediction - cmd), 0)
  loss = tf.reduce_mean(tf.abs(prediction - cmd))

  return loss, loss_split
