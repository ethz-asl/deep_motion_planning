
import logging

import tensorflow as tf

NAME = "two_fully_connected"

INPUT_SIZE = 723
CMD_SIZE = 2

HIDDEN_1 = 600
HIDDEN_2 = 500
HIDDEN_3 = 250

def learning_rate(initial):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(initial, global_step, 
            250000, 0.85, staircase=True)
    return global_step, learning_rate

def inference(data):
    weights_hidden1 = tf.get_variable("weights_hidden1", shape=[INPUT_SIZE, HIDDEN_1],
                       initializer=tf.contrib.layers.xavier_initializer())
    biases_hidden1 = tf.Variable(tf.zeros([HIDDEN_1]))
    weights_hidden2 = tf.get_variable("weights_hidden2", shape=[HIDDEN_1, HIDDEN_2],
                       initializer=tf.contrib.layers.xavier_initializer())
    biases_hidden2 = tf.Variable(tf.zeros([HIDDEN_2]))
    weights_hidden3 = tf.get_variable("weights_hidden3", shape=[HIDDEN_2, HIDDEN_3],
                       initializer=tf.contrib.layers.xavier_initializer())
    biases_hidden3 = tf.Variable(tf.zeros([HIDDEN_3]))
    weights_out = tf.Variable(
        tf.truncated_normal([HIDDEN_3, CMD_SIZE], stddev=0.1) / tf.sqrt(tf.to_float(HIDDEN_3)))
    biases_out = tf.Variable(tf.zeros([CMD_SIZE]))

    hidden_1 = tf.nn.relu(tf.add(tf.matmul(data, weights_hidden1), biases_hidden1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights_hidden2), biases_hidden2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, weights_hidden3), biases_hidden3))
    prediction = tf.add(tf.matmul(hidden_3, weights_out), biases_out, name='prediction')

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
