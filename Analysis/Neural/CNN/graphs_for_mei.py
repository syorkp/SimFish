import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from Analysis.load_data import load_data

tf.logging.set_verbosity(tf.logging.ERROR)


class MEICore:
    """For all observations - just predicts nonlinear features of these - can reflect right observation and reuse
    this network."""

    def __init__(self, my_scope="MEICore"):
        # Observation (from one eye)
        self.observation = tf.placeholder(shape=[None, 100, 3], dtype=tf.float32, name='obs')
        # self.reshaped_observation = tf.reshape(self.observation, shape=[-1, input_shape[0], input_shape[1]],
        #                                        name="reshaped_observation")

        # TODO: get their hyperparameters.
        self.conv1l = tf.layers.conv1d(inputs=self.observation, filters=16, kernel_size=16, strides=4,
                                       padding='valid', activation=tf.nn.elu, name=my_scope + '_conv1l',
                                       use_bias=False)
        self.conv2l = tf.layers.conv1d(inputs=self.conv1l, filters=8, kernel_size=8, strides=2, padding='valid',
                                       activation=tf.nn.elu, name=my_scope + '_conv2l', use_bias=False)
        self.conv3l = tf.layers.conv1d(inputs=self.conv2l, filters=8, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.elu, name=my_scope + '_conv3l', use_bias=False)

        self.output = self.conv3l


class MEIReadout:

    def __init__(self, readout_input, my_scope="MEIReadout"):
        flattened_readout_input = tf.layers.flatten(readout_input)

        self.predicted_neural_activity = tf.layers.dense(flattened_readout_input, 1, activation=tf.nn.elu,
                                                         kernel_initializer=tf.orthogonal_initializer,
                                                         name=my_scope + "_readout_dense", trainable=True)


class Trainer:

    def __init__(self, predicted_responses, learning_rate=0.01, max_gradient_norm=1.5):
        self.actual_responses = tf.placeholder(shape=[None], dtype=float)
        self.total_loss = tf.squared_difference(predicted_responses, self.actual_responses)

        self.model_params = tf.trainable_variables()
        self.model_gradients = tf.gradients(self.total_loss, self.model_params)
        self.model_gradients, _grad_norm = tf.clip_by_global_norm(self.model_gradients, max_gradient_norm)
        self.model_gradients = list(zip(self.model_gradients, self.model_params))

        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
        self.train = self.trainer.apply_gradients(self.model_gradients)
