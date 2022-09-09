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
        self.dense_layer = tf.layers.dense(flattened_readout_input, 1, activation=tf.nn.elu,
                                           kernel_initializer=tf.orthogonal_initializer,
                                           name=my_scope + "_readout_dense", trainable=True)
        self.predicted_neural_activity = tf.add(self.dense_layer, 1)


class Trainer:

    def __init__(self, predicted_responses, learning_rate=0.01, max_gradient_norm=1.5):
        self.actual_responses = tf.placeholder(shape=[None], dtype=float)
        self.total_loss = tf.reduce_mean(tf.squared_difference(predicted_responses, self.actual_responses))

        self.model_params = tf.trainable_variables()
        self.model_gradients = tf.gradients(self.total_loss, self.model_params)
        self.model_gradients, _grad_norm = tf.clip_by_global_norm(self.model_gradients, max_gradient_norm)
        self.model_gradients = list(zip(self.model_gradients, self.model_params))

        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
        self.train = self.trainer.apply_gradients(self.model_gradients)


class TrainerExtended:

    def __init__(self, predicted_responses, n_units, learning_rate=0.01, max_gradient_norm=1.5):
        self.actual_responses = tf.placeholder(shape=[None, n_units], dtype=float)
        # MSE loss
        self.total_loss = tf.reduce_mean(tf.squared_difference(predicted_responses, self.actual_responses))
        # Poisson loss
        # self.total_loss = tf.reduce_mean(predicted_responses-(self.actual_responses * tf.log(predicted_responses + 0.000001)))

        self.model_params = tf.trainable_variables()

        targeted_params = [param for param in self.model_params if "conv1l" in param.name]
        targeted_params_shapes = [param.shape[0] for param in targeted_params]
        indices = [tf.range(0, shape, dtype=tf.float32) for shape in targeted_params_shapes]
        distance_from_centre = tf.abs([indices_s-targeted_params_shapes[j] for j, indices_s in enumerate(indices)])
        gaussian_operations = 1 * (1 - tf.exp(-(1/2*(0.5**2))*distance_from_centre))
        self.lossL2 = tf.reduce_mean(tf.add_n([tf.math.square(param) * gaussian_operations[i] for i, param in enumerate(targeted_params)]))

        # self.lossL2 = [tf.nn.l2_loss(param) for param in self.model_params if "conv1l" in param.name] * 0.001
        # self.total_loss = tf.add(self.total_loss, self.lossL2)

        self.model_gradients = tf.gradients(self.total_loss, self.model_params)
        self.model_gradients, _grad_norm = tf.clip_by_global_norm(self.model_gradients, max_gradient_norm)
        self.model_gradients = list(zip(self.model_gradients, self.model_params))

        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
        self.train = self.trainer.apply_gradients(self.model_gradients)

