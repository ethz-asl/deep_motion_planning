
import logging

import tensorflow as tf

NAME = "two_fully_connected"

INPUT_SIZE = 723
CMD_SIZE = 2

HIDDEN_1 = 500

def inference(data):
    weights_hidden = tf.Variable(
        tf.truncated_normal([INPUT_SIZE, HIDDEN_1], stddev=0.1))
    biases_hidden = tf.Variable(tf.zeros([HIDDEN_1]))
    weights_out = tf.Variable(
        tf.truncated_normal([HIDDEN_1, CMD_SIZE], stddev=0.1))
    biases_out = tf.Variable(tf.zeros([CMD_SIZE]))
    hidden_1 = tf.nn.tanh(tf.matmul(data, weights_hidden) + biases_hidden)
    prediction = tf.nn.tanh(tf.matmul(hidden_1, weights_out) + biases_out)

    return prediction

def loss(prediction, cmd):
    loss = tf.reduce_sum(tf.abs(prediction - cmd), 0)

    return loss

def training(loss, learning_rate):
    tf.scalar_summary('loss_linear_x', loss[0])
    tf.scalar_summary('loss_angular_yaw', loss[1])
    tf.scalar_summary('learning_rate', learning_rate)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(prediction, labels):
    return tf.reduce_sum(tf.abs(prediction - labels), 0)
