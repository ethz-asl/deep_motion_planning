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
NAME = 'convolutional'

# The size of the input layer
INPUT_SIZE = 1083
# The size of the output layer
CMD_SIZE = 2

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

def inference(data, keep_prob, sample_size, training=True, reuse=False, output_name='prediction'):
  """
  Define the deep neural network used for inference
  """
  # slice 709 elements to get the correct size for the next convolutions
  laser = tf.slice(data, [0,0], [sample_size,1080])
  goal = tf.slice(data, [0,1080], [sample_size,3])

  laser = tf.reshape(laser, [sample_size, 1, 1080, 1])
  hidden_1 = conv2d(laser, 64, [1,7], stride=3, normalizer_fn=batch_norm,
      weights_initializer=xavier_initializer_conv2d(),
      weights_regularizer=l1_regularizer(0.001), reuse=reuse, trainable=training, scope='layer_scope_1')
  hidden_1 = contrib.layers.max_pool2d(hidden_1, [1,3],[1,3], 'SAME')
  hidden_2 = conv2d(hidden_1, 64, [1,3], normalizer_fn=batch_norm,
      weights_initializer=xavier_initializer_conv2d(),
      weights_regularizer=l1_regularizer(0.001), reuse=reuse, trainable=training, scope='layer_scope_2')
  hidden_3 = conv2d(hidden_2, 64, [1,3], activation_fn=None, normalizer_fn=batch_norm,
      weights_initializer=xavier_initializer_conv2d(),
      weights_regularizer=l1_regularizer(0.001), reuse=reuse, trainable=training, scope='layer_scope_3')
  hidden_3 = tf.nn.relu(hidden_3 + hidden_1)
  hidden_4 = conv2d(hidden_3, 64, [1,3], normalizer_fn=batch_norm,
      weights_initializer=xavier_initializer_conv2d(),
      weights_regularizer=l1_regularizer(0.001), reuse=reuse, trainable=training, scope='layer_scope_4')
  hidden_5 = conv2d(hidden_4, 64, [1,3], activation_fn=None, normalizer_fn=batch_norm,
      weights_initializer=xavier_initializer_conv2d(),
      weights_regularizer=l1_regularizer(0.001), reuse=reuse, trainable=training, scope='layer_scope_5')
  hidden_5 = tf.nn.relu(hidden_5 + hidden_3)

  pooling = contrib.layers.avg_pool2d(hidden_5, [1,3],[1,3], 'SAME')
  pooling = contrib.layers.flatten(pooling)
  combined = tf.concat([pooling, goal], axis=1)
  fc_5 = fully_connected(combined, 1024, weights_initializer=xavier_initializer(),
      weights_regularizer=l1_regularizer(0.001), reuse=reuse, trainable=training, scope='fc_scope_5')
  fc_6 = fully_connected(fc_5, 1024, weights_initializer=xavier_initializer(),
      weights_regularizer=l1_regularizer(0.001), reuse=reuse, trainable=training, scope='fc_scope_6')
  fc_7 = fully_connected(fc_6, 512, weights_initializer=xavier_initializer(),
      weights_regularizer=l1_regularizer(0.001), reuse=reuse, trainable=training, scope='fc_scope_7')
  prediction = fully_connected(fc_7, CMD_SIZE, activation_fn=None, reuse=reuse, trainable=training, scope='layer_scope_pred')

  prediction = tf.identity(prediction, name=output_name)

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
