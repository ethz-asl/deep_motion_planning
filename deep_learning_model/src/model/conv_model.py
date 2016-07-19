"""
This file contains methods that describe a Tensorflow model.
It can be used as template for a completely new model and is imported
in the training script
"""
import logging

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, conv2d

# Give the model a descriptive name
NAME = 'convolutional'

# The size of the input layer
INPUT_SIZE = 723
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

def inference(data, keep_prob, reuse=False, output_name='prediction'):
    """
    Define the deep neural network used for inference
    """

    laser = tf.slice(data, [0,0], [-1,720])
    goal = tf.slice(data, [0,720], [-1,-1])

    laser = tf.reshape(laser, [tf.shape(laser)[0], 1, tf.shape(laser)[1], 1])
    hidden_1 = conv2d(laser, 25, [1,3], reuse=reuse, scope='layer_scope_1')
    hidden_2 = conv2d(hidden_1, 50, [1,3], reuse=reuse, scope='layer_scope_2')
    hidden_3 = conv2d(hidden_2, 100, [1,3], reuse=reuse, scope='layer_scope_3')

    pooling = tf.nn.avg_pool(hidden_3, [1,1,4,4],[1,1,1,1])
    pooling = tf.reshape(pooling, [tf.shape(pooling)[0], 180*25])
    combined = tf.concat(1,[pooling, goal])
    fc_7 = fully_connected(combined, 1024, reuse=reuse, scope='layers_scope_4')
    prediction = fully_connected(fc_7, CMD_SIZE, activation_fn=None, reuse=reuse, scope='layer_scope_pred')

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
