import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class BaseNetwork:

    """A base network containing all components used in child networks."""

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim, new_simulation):

        self.num_arms = simulation.fish.left_eye.observation_size  # Rays for each eye
        self.rnn_dim = rnn_dim

        self.train_length = tf.placeholder(dtype=tf.int32, shape=[], name="train_length")
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')

        self.rnn_state_in = rnn_cell.zero_state(self.train_length, tf.float32)
        self.rnn_state_in_ref = rnn_cell.zero_state(self.train_length, tf.float32)

        # Networks Inputs
        self.prev_actions = tf.placeholder(shape=[None, action_dim], dtype=tf.float32, name='prev_actions')
        self.internal_state = tf.placeholder(shape=[None, internal_states], dtype=tf.float32, name='internal_state')

        self.observation = tf.placeholder(shape=[None, 3, 2], dtype=tf.float32, name='obs')
        self.reshaped_observation = tf.reshape(self.observation, shape=[-1, self.num_arms, 3, 2],
                                               name="reshaped_observation")

        #            ----------        Non-Reflected       ---------            #

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
            [self.conv4l_flat, self.conv4r_flat, self.prev_actions, self.internal_state], 1, name="flattened_conv")

        # Recurrent Layer
        self.rnn_in = tf.layers.dense(self.conv_with_states, self.rnn_dim, activation=tf.nn.relu,
                                      kernel_initializer=tf.orthogonal_initializer,
                                      trainable=True, name=my_scope + '_rnn_in')
        self.convFlat = tf.reshape(self.rnn_in, [self.batch_size, self.train_length, self.rnn_dim],
                                   name="flattened_shared_rnn_input")

        self.rnn, self.rnn_state_shared = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell,
                                                            dtype=tf.float32,
                                                            initial_state=self.rnn_state_in, scope=my_scope + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, self.rnn_dim], name="shared_rnn_output")
        self.rnn_output = self.rnn

        #            ----------        Reflected       ---------            #

        self.ref_left_eye = tf.reverse(self.right_eye, [1])
        self.ref_right_eye = tf.reverse(self.left_eye, [1])

        self.conv1l_ref = tf.layers.conv1d(inputs=self.ref_left_eye, filters=16, kernel_size=16, strides=4,
                                           padding='valid',
                                           activation=tf.nn.relu, name=my_scope + '_conv1l', reuse=True)
        self.conv2l_ref = tf.layers.conv1d(inputs=self.conv1l_ref, filters=8, kernel_size=8, strides=2,
                                           padding='valid',
                                           activation=tf.nn.relu, name=my_scope + '_conv2l', reuse=True)
        self.conv3l_ref = tf.layers.conv1d(inputs=self.conv2l_ref, filters=8, kernel_size=4, strides=1, padding='valid',
                                           activation=tf.nn.relu, name=my_scope + '_conv3l', reuse=True)
        self.conv4l_ref = tf.layers.conv1d(inputs=self.conv3l_ref, filters=64, kernel_size=4, strides=1,
                                           padding='valid',
                                           activation=tf.nn.relu, name=my_scope + '_conv4l', reuse=True)

        self.conv1r_ref = tf.layers.conv1d(inputs=self.ref_right_eye, filters=16, kernel_size=16, strides=4,
                                           padding='valid',
                                           activation=tf.nn.relu, name=my_scope + '_conv1r', reuse=True)
        self.conv2r_ref = tf.layers.conv1d(inputs=self.conv1r_ref, filters=8, kernel_size=8, strides=2, padding='valid',
                                           activation=tf.nn.relu, name=my_scope + '_conv2r', reuse=True)
        self.conv3r_ref = tf.layers.conv1d(inputs=self.conv2r_ref, filters=8, kernel_size=4, strides=1, padding='valid',
                                           activation=tf.nn.relu, name=my_scope + '_conv3r', reuse=True)
        self.conv4r_ref = tf.layers.conv1d(inputs=self.conv3r_ref, filters=64, kernel_size=4, strides=1,
                                           padding='valid',
                                           activation=tf.nn.relu, name=my_scope + '_conv4r', reuse=True)

        self.conv4l_flat_ref = tf.layers.flatten(self.conv4l_ref)
        self.conv4r_flat_ref = tf.layers.flatten(self.conv4r_ref)
        # self.prev_actions_ref = tf.reverse(self.prev_actions, [1])
        # self.internal_state_ref = tf.reverse(self.internal_state, [1])
        self.prev_actions_ref = self.prev_actions
        self.internal_state_ref = self.internal_state

        self.conv_with_states_ref = tf.concat(
            [self.conv4l_flat_ref, self.conv4r_flat_ref, self.prev_actions_ref, self.internal_state_ref], 1)
        self.rnn_in_ref = tf.layers.dense(self.conv_with_states_ref, self.rnn_dim, activation=tf.nn.relu,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          trainable=True, name=my_scope + '_rnn_in', reuse=True)
        self.convFlat_ref = tf.reshape(self.rnn_in_ref, [self.batch_size, self.train_length, self.rnn_dim])

        self.rnn_ref, self.rnn_state_ref = tf.nn.dynamic_rnn(inputs=self.convFlat_ref, cell=rnn_cell,
                                                             dtype=tf.float32,
                                                             initial_state=self.rnn_state_in_ref,
                                                             scope=my_scope + '_rnn')  # No need to reuse as takes rnn_cell as argument for both.
        self.rnn_ref = tf.reshape(self.rnn_ref, shape=[-1, self.rnn_dim])
        self.rnn_output_ref = self.rnn_ref

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower

