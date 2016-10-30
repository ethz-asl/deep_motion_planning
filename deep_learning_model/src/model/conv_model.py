"""
This file contains methods that describe a Tensorflow model.
It can be used as template for a completely new model and is imported
in the training script
"""
import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim

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
                        initializer=slim.xavier_initializer()),
                        tf.get_variable('biases_hidden{}'.format(index), [output_size]))

def res_block(x, scope_number, reuse, training):
    """
    Add a residual block (with new design) to the Tensorflow graph
    """
    with tf.name_scope('residual_block_{}'.format(scope_number)) as scope:
        net = tf.nn.relu(slim.batch_norm(x, is_training=True, reuse=reuse,
                scope='resnet_batch_{}_1'.format(scope_number)))
        net = slim.conv2d(net, 64, [1,3], stride=1, activation_fn=None,
                weights_initializer=slim.xavier_initializer_conv2d(), 
                weights_regularizer=slim.l2_regularizer(0.001),
                reuse=reuse, trainable=training, scope='residual_block_{}_1'.format(scope_number))
        net = tf.nn.relu(slim.batch_norm(net, is_training=True, reuse=reuse,
            scope='residual_block_{}_2'.format(scope_number)))
        net = slim.conv2d(net, 64, [1,3], stride=1, activation_fn=None,
                weights_initializer=slim.xavier_initializer_conv2d(), 
                weights_regularizer=slim.l2_regularizer(0.001),
                reuse=reuse, trainable=training, scope='residual_block_{}_2'.format(scope_number))
    return tf.add(x,net)

def inference(data, keep_prob, sample_size, training=True, reuse=False, output_name='prediction'):
    """
    Define the deep neural network used for inference
    """

    # slice 709 elements to get the correct size for the next convolutions
    laser = tf.slice(data, [0,0], [sample_size,1080])
    goal = tf.slice(data, [0,1080], [sample_size,3])

    laser = tf.reshape(laser, [sample_size, 1, 1080, 1])
    net = slim.conv2d(laser, 64, [1,3], stride=1, normalizer_fn=None,
            weights_initializer=slim.xavier_initializer_conv2d(), 
            weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse, trainable=training, scope='layer_scope_1')
    net = slim.max_pool2d(net, [1,3],[1,3], 'SAME')
    net = slim.conv2d(net, 64, [1,3], stride=1, normalizer_fn=None, activation_fn=None,
            weights_initializer=slim.xavier_initializer_conv2d(), 
            weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse, trainable=training, scope='layer_scope_2')
    net = slim.max_pool2d(net, [1,3],[1,3], 'SAME')
    
    net = res_block(net, 3, reuse=reuse, training=training)
    net = res_block(net, 4, reuse=reuse, training=training)
    net = res_block(net, 5, reuse=reuse, training=training)

    pooling = slim.avg_pool2d(net, [1,3],[1,3], 'SAME')
    pooling = slim.flatten(tf.nn.dropout(pooling, keep_prob))
    combined = tf.concat(1,[pooling, goal])

    # Attention
    att = slim.fully_connected(combined, 2560, weights_initializer=slim.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse, trainable=training, scope='att_scope_2')
    att = tf.nn.dropout(att, keep_prob)
    alpha = slim.fully_connected(att, 2560, activation_fn=None, reuse=reuse, trainable=training,
            scope='alpha_scope')
    alpha = tf.nn.softmax(alpha)

    weighted_sensor = tf.mul(pooling, alpha)
    combined = tf.concat(1,[weighted_sensor, goal])

    net = slim.fully_connected(combined, 2048, weights_initializer=slim.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse, trainable=training, scope='fc_scope_5')
    net = tf.nn.dropout(net, keep_prob)
    net = slim.fully_connected(net, 2048, weights_initializer=slim.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse, trainable=training, scope='fc_scope_6')
    prediction = slim.fully_connected(net, CMD_SIZE, activation_fn=None, reuse=reuse, trainable=training, scope='layer_scope_pred')

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
