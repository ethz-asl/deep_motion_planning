
import logging

import tensorflow as tf

NAME = "two_fully_connected"

INPUT_SIZE = 723
CMD_SIZE = 2

HIDDEN_1 = 500
HIDDEN_2 = 250

def learning_rate(initial):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(initial, global_step, 
            100000, 0.96, staircase=True)
    return global_step, learning_rate

def inference(data):
    weights_hidden = tf.Variable(
        tf.truncated_normal([INPUT_SIZE, HIDDEN_1], stddev=0.1) /
        tf.sqrt(tf.to_float(INPUT_SIZE)))
    biases_hidden = tf.Variable(tf.zeros([HIDDEN_1]))
    weights_out = tf.Variable(
        tf.truncated_normal([HIDDEN_1, CMD_SIZE], stddev=0.1) / tf.sqrt(tf.to_float(HIDDEN_1)))
    biases_out = tf.Variable(tf.zeros([CMD_SIZE]))
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(data, weights_hidden), biases_hidden))
    prediction = tf.add(tf.matmul(hidden_1, weights_out), biases_out)

    return prediction

def loss(prediction, cmd):
    loss_split = tf.reduce_mean((prediction - cmd), 0)
    loss = tf.reduce_mean(tf.abs(prediction - cmd))

    return loss, loss_split

def training(loss, loss_split, learning_rate, global_step):
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(prediction, cmd):
    loss_split = tf.reduce_mean((prediction - cmd), 0)
    loss = tf.reduce_mean(tf.abs(prediction - cmd))

    return loss, loss_split
