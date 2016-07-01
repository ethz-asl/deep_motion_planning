"""
This file contains methods that describe a Tensorflow model.
It can be used as template for a completely new model and is imported
in the training script
"""
import logging

import tensorflow as tf

# Give the model a descriptive name
NAME = 'two_fully_connected'

# The size of the input layer
INPUT_SIZE = 723
# The size of the output layer
CMD_SIZE = 2

# Place some generic parameters here
HIDDEN_1 = 723
HIDDEN_2 = 723
HIDDEN_3 = 500

def learning_rate(initial):
    """
    Define learning rate for your model
    """
    # We use an exponential decy for the model
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(initial, global_step, 
            250000, 0.85, staircase=True)
    return global_step, learning_rate

def inference(data, reuse=False, output_name='prediction'):
    """
    Define the deep neural network used for inference
    """
    with tf.variable_scope("prediction_scope") as scope:
        if reuse:
            scope.reuse_variables()
            
        weights_hidden1 = tf.get_variable('weights_hidden1', shape=[INPUT_SIZE, HIDDEN_1],
                        initializer=tf.contrib.layers.xavier_initializer())
        biases_hidden1 = tf.get_variable('biases_hidden1', [HIDDEN_1],
                        initializer=tf.constant_initializer(0.0))
        weights_hidden2 = tf.get_variable('weights_hidden2', shape=[HIDDEN_1, HIDDEN_2],
                        initializer=tf.contrib.layers.xavier_initializer())
        biases_hidden2 = tf.get_variable('biases_hidden2', [HIDDEN_2],
                        initializer=tf.constant_initializer(0.0))
        weights_hidden3 = tf.get_variable('weights_hidden3', shape=[HIDDEN_2, HIDDEN_3],
                        initializer=tf.contrib.layers.xavier_initializer())
        biases_hidden3 = tf.get_variable('biases_hidden3', [HIDDEN_3],
                        initializer=tf.constant_initializer(0.0))
        weights_out = tf.get_variable('weights_out', [HIDDEN_3, CMD_SIZE],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases_out = tf.get_variable('biases_out', [CMD_SIZE],
                        initializer=tf.constant_initializer(0.0))

    hidden_1 = tf.nn.relu(tf.add(tf.matmul(data, weights_hidden1), biases_hidden1))
    hidden_2 = tf.nn.relu(tf.add(tf.add(tf.matmul(hidden_1, weights_hidden2), biases_hidden2), data))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, weights_hidden3), biases_hidden3))
    prediction = tf.add(tf.matmul(hidden_3, weights_out), biases_out, name=output_name)

    return prediction

def loss(prediction, cmd):
    """
    Define the loss used during the training steps
    """
    loss_split = tf.reduce_mean((prediction - cmd), 0)
    loss = tf.reduce_mean(tf.abs(prediction - cmd))

    return loss, loss_split

def training(loss, loss_split, learning_rate, global_step):
    """
    Perform a single training step optimization
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(prediction, cmd):
    """
    Define how to evaluate the model
    """
    loss_split = tf.reduce_mean((prediction - cmd), 0)
    loss = tf.reduce_mean(tf.abs(prediction - cmd))

    return loss, loss_split
