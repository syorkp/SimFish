import sklearn.preprocessing
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class A2CNetwork:

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states=2, actor_learning_rate_impulse=0.00001,
                 actor_learning_rate_angle=0.00001, critic_learning_rate=0.00056, max_impulse=80, max_angle_change=1):

        # Variables
        self.num_arms = len(simulation.fish.left_eye.vis_angles)  # Rays for each eye
        self.rnn_dim = rnn_dim
        self.rnn_output_size = self.rnn_dim

        self.trainLength = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.init_xavier = tf.keras.initializers.glorot_normal()

        # Network Inputs
        # self.actions = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='actions')
        self.prev_actions = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='prev_actions')
        self.internal_state = tf.placeholder(shape=[None, internal_states], dtype=tf.float32, name='internal_state')

        self.observation = tf.placeholder(shape=[None, 3, 2], dtype=tf.float32, name='obs')
        self.reshaped_observation = tf.reshape(self.observation, shape=[-1, self.num_arms, 3, 2])
        self.left_eye = self.reshaped_observation[:, :, :, 0]
        self.right_eye = self.reshaped_observation[:, :, :, 1]

        # Convolutional Layers
        self.conv1l = tf.layers.conv1d(inputs=self.left_eye, filters=16, kernel_size=16, strides=4, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv1l')
        self.conv2l = tf.layers.conv1d(inputs=self.conv1l, filters=8, kernel_size=8, strides=2, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv2l')
        self.conv3l = tf.layers.conv1d(inputs=self.conv2l, filters=8, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv3l')
        self.conv4l = tf.layers.conv1d(inputs=self.conv3l, filters=64, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv4l')

        self.conv1r = tf.layers.conv1d(inputs=self.right_eye, filters=16, kernel_size=16, strides=4, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv1r')
        self.conv2r = tf.layers.conv1d(inputs=self.conv1r, filters=8, kernel_size=8, strides=2, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv2r')
        self.conv3r = tf.layers.conv1d(inputs=self.conv2r, filters=8, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv3r')
        self.conv4r = tf.layers.conv1d(inputs=self.conv3r, filters=64, kernel_size=4, strides=1, padding='valid',
                                       activation=tf.nn.relu, name=my_scope + '_conv4r')

        self.conv4l_flat = tf.layers.flatten(self.conv4l)
        self.conv4r_flat = tf.layers.flatten(self.conv4r)

        self.conv_with_states = tf.concat(
            [self.conv4l_flat, self.conv4r_flat, self.prev_actions, self.internal_state], 1)

        # Recurrent Layer
        self.rnn_in = tf.layers.dense(self.conv_with_states, self.rnn_dim, activation=tf.nn.relu,
                                      kernel_initializer=tf.orthogonal_initializer,
                                      trainable=True, name=my_scope + '_rnn_in')
        self.convFlat = tf.reshape(self.rnn_in, [self.batch_size, self.trainLength, self.rnn_dim])

        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32,
                                                     initial_state=self.state_in, scope=my_scope + '_rnn',)
        self.rnn = tf.reshape(self.rnn, shape=[-1, self.rnn_dim])
        self.rnn_output = self.rnn
        self.rnn_state2 = self.rnn_state

        # Critic (value) output
        self.Value = tf.layers.dense(self.rnn_output, 1, None, self.init_xavier)

        # Actor impulse output
        self.mu_impulse = tf.layers.dense(self.rnn_output, 1, None, self.init_xavier)
        self.mu_impulse = tf.math.abs(self.mu_impulse)

        self.sigma_impulse = tf.layers.dense(self.rnn_output, 1, None, self.init_xavier)
        self.sigma_impulse = tf.math.abs(self.sigma_impulse)

        self.norm_dist_impulse = tf.distributions.Normal(self.mu_impulse, self.sigma_impulse)
        self.action_tf_var_impulse = tf.squeeze(self.norm_dist_impulse.sample(1), axis=0)
        self.action_tf_var_impulse = tf.math.abs(self.action_tf_var_impulse)
        self.action_tf_var_impulse = tf.clip_by_value(self.action_tf_var_impulse, 0, 1)
        self.action_tf_var_impulse = tf.math.multiply(self.action_tf_var_impulse, max_impulse)

        # Actor angle output
        self.mu_angle = tf.layers.dense(self.rnn_output, 1, None, self.init_xavier)
        self.sigma_angle = tf.layers.dense(self.rnn_output, 1, None, self.init_xavier)
        self.sigma_angle = tf.math.abs(self.sigma_angle)
        self.norm_dist_angle = tf.distributions.Normal(self.mu_angle, self.sigma_angle)
        self.action_tf_var_angle = tf.squeeze(self.norm_dist_angle.sample(1), axis=0)
        self.action_tf_var_angle = tf.clip_by_value(self.action_tf_var_angle, -1, 1)
        self.action_tf_var_angle = tf.math.multiply(self.action_tf_var_angle, max_angle_change)

        self.action_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='action_placeholder')
        self.delta_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='delta')
        self.target_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='target')

        # Actor (policy) loss function - Impulse
        self.loss_actor_impulse = -tf.log(self.norm_dist_impulse.prob(tf.math.divide(self.action_placeholder[0][0], max_impulse)) + 1e-5) * self.delta_placeholder
        self.training_op_actor_impulse = tf.train.AdamOptimizer(actor_learning_rate_impulse, name='actor_optimizer_impulse').minimize(self.loss_actor_impulse)

        # Actor (policy) loss function - Angle
        self.loss_actor_angle = -tf.log(self.norm_dist_impulse.prob(tf.math.divide(self.action_placeholder[0][1], max_angle_change)) + 1e-5) * self.delta_placeholder
        self.training_op_actor_angle = tf.train.AdamOptimizer(actor_learning_rate_angle, name='actor_optimizer_angle').minimize(self.loss_actor_angle)

        # Critic (state-value) loss function
        self.loss_critic = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.Value), self.target_placeholder))
        self.training_op_critic = tf.train.AdamOptimizer(critic_learning_rate, name='critic_optimizer').minimize(self.loss_critic)


