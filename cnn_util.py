# ========================================
# [] File Name : cnn_util.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# ========================================

import tensorflow as tf


def conv_bn_sc_relu(inputs, filters, k_size, stride, padding, scope_name):

    with tf.variable_scope(scope_name, reuse=True) as scope:

        in_channels = inputs.shape[-1]
    
        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters], initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [filters], initializer=tf.random_normal_initializer())

        conv = tf.nn.conv2d(input=inputs, filter=kernel, strides=[1, stride, stride, 1], padding=padding, use_cudnn_on_gpu=True)
    
        # Perform a batch normalization
        norm = tf.layers.batch_normalization(inputs=conv)

        # Scale the normalized batch
        # scaled_batch = scale(inputs=norm, scope_name='batch_norm_scaler')

    # Perform a relu and return
    # return tf.nn.relu(scaled_batch + biases, name=scope.name)
    return tf.nn.relu(norm + biases, name=scope.name)


def maxpool(inputs, k_size, stride, padding, scope_name):

    with tf.variable_scope(scope_name, reuse=True) as scope:
        pool = tf.nn.max_pool(value=inputs, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding=padding)
        
    return pool

def scale(inputs, scope_name):

    with tf.variable_scope(scope_name, reuse=True) as scope:

        in_dim = inputs.shape[-1]
        alpha = tf.get_variable(name='alpha', shape=(in_dim, ), trainable=True)
        beta = tf.get_variable(name='beta', shape=(in_dim, ), trainable=True)

        scaled_input = alpha * input + beta

    return scaled_input

def saf_pool(inputs, k_size, stride, padding, scope_name):

    with tf.variable_scope(scope_name, tf.True) as scope:

        # Perform a Maxpool
        pooled = maxpool(
            inputs=inputs,
            k_size=k_size,
            stride=stride,
            padding=padding,
            scope_name=scope_name
        )

        # Perform a Dropout
        saf_pooled = tf.nn.dropout(
            x=pooled,
            keep_prob=0.8
        )

    return saf_pooled



