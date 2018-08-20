import tensorflow as tf


def conv_relu(inputs, filters, k_size, stride, padding, scope_name):

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        in_channels = inputs.shape[-1]

        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters], initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [filters], initializer=tf.random_normal_initializer())

        conv = tf.nn.conv2d(input=inputs, filter=kernel, strides=[1, stride, stride, 1], padding=padding, use_cudnn_on_gpu=True)
    
    return tf.nn.relu(conv + biases, name=scope.name)

def maxpool(inputs, k_size, stride, padding, scope_name):

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(input=inputs, filter=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding=padding)
        
    return pool


