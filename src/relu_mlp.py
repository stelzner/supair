# License: MIT
# Author: Robert Peharz

import tensorflow as tf


class ReluMLP(object):
    def __init__(self, num_inputs, num_units, act_types=None, xavier=False):
        self.num_inputs = num_inputs
        self.num_units = num_units

        self.act_types = act_types

        self.layers = []
        self.weights = []
        self.biases = []
        self.xavier = xavier

        self.make_network()

    def forward(self, inputs):
        prev_layer = inputs
        for k in range(len(self.num_units)):
            activation = tf.matmul(prev_layer, self.weights[k]) + self.biases[k]

            if self.act_types is None:
                layer = tf.nn.relu(activation)
            elif self.act_types[k] == 's':
                layer = tf.sigmoid(activation)
            elif self.act_types[k] == 'r':
                layer = tf.nn.relu(activation)
            elif self.act_types[k] == 'l':
                layer = activation
            else:
                raise NotImplementedError('unknown activation function')
            prev_layer = layer
        return prev_layer

    def make_network(self):
        for k in range(len(self.num_units)):
            num_units_cur = self.num_units[k]
            num_units_prev = self.num_inputs if k == 0 else self.num_units[k - 1]

            with tf.variable_scope('layer_{}'.format(k)):
                if self.xavier:
                    W = tf.get_variable(
                        'W',
                        [num_units_prev, num_units_cur],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32)
                else:
                    W = tf.get_variable(
                        'W',
                        [num_units_prev, num_units_cur],
                        initializer=tf.truncated_normal_initializer(stddev=0.5, dtype=tf.float32),
                        dtype=tf.float32)

                b = tf.get_variable(
                    'b',
                    [num_units_cur],
                    initializer=tf.constant_initializer(0.1),
                    dtype=tf.float32)

                self.weights.append(W)
                self.biases.append(b)
