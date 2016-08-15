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
INPUT_SIZE = 543
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
    with tf.variable_scope("prediction_scope") as scope:
        if reuse:
            scope.reuse_variables()
            
        weights_hidden_6, biases_hidden_6 = __get_variable__(6, INPUT_SIZE, 2048)
        weights_hidden_7, biases_hidden_7 = __get_variable__(7, 2048, 2048)
        weights_hidden_8, biases_hidden_8 = __get_variable__(8, 2048, 1024)
        weights_out = tf.get_variable('weights_out', [1024, CMD_SIZE],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases_out = tf.get_variable('biases_out', [CMD_SIZE],
                        initializer=tf.constant_initializer(0.0))

    hidden_6 = tf.nn.relu(tf.add(tf.matmul(data, weights_hidden_6), biases_hidden_6))
    hidden_7 = tf.nn.relu(tf.add(tf.matmul(hidden_6, weights_hidden_7), biases_hidden_7))
    hidden_7 = tf.nn.dropout(hidden_7, keep_prob)
    hidden_8 = tf.nn.relu(tf.add(tf.matmul(hidden_7, weights_hidden_8), biases_hidden_8))
    hidden_8 = tf.nn.dropout(hidden_8, keep_prob)
    prediction = tf.add(tf.matmul(hidden_8, weights_out), biases_out, name=output_name)

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
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(prediction, cmd):
    """
    Define how to evaluate the model
    """
    loss_split = tf.reduce_mean((prediction - cmd), 0)
    loss = tf.reduce_mean(tf.abs(prediction - cmd))

    return loss, loss_split
