import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class QNetwork:

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, num_actions, internal_states=2, learning_rate=0.0001,
                 extra_layer=False):
        """The network receives the observation from both eyes, processes it
        #through convolutional layers, concatenates it with the internal state
        #and feeds it to the RNN."""

        self.num_arms = len(simulation.fish.left_eye.vis_angles)  # Rays for each eye
        self.rnn_dim = rnn_dim
        self.rnn_output_size = self.rnn_dim
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.actions_one_hot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_actions')
        self.prev_actions_one_hot = tf.one_hot(self.prev_actions, num_actions, dtype=tf.float32)

        self.internal_state = tf.placeholder(shape=[None, internal_states], dtype=tf.float32, name='internal_state')

        self.observation = tf.placeholder(shape=[None, 3, 2], dtype=tf.float32, name='obs')
        self.reshaped_observation = tf.reshape(self.observation, shape=[-1, self.num_arms, 3, 2])
        self.left_eye = self.reshaped_observation[:, :, :, 0]
        self.right_eye = self.reshaped_observation[:, :, :, 1]

        #                ------------ Common to Both ------------                   #

        self.exp_keep = tf.placeholder(shape=None, dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None, dtype=tf.float32)
        self.trainLength = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

        #                ------------ Normal network ------------                   #

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

        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levels.

        self.conv4l_flat = tf.layers.flatten(self.conv4l)
        self.conv4r_flat = tf.layers.flatten(self.conv4r)

        self.conv_with_states = tf.concat(
            [self.conv4l_flat, self.conv4r_flat, self.prev_actions_one_hot, self.internal_state], 1)
        self.rnn_in = tf.layers.dense(self.conv_with_states, self.rnn_dim, activation=tf.nn.relu,
                                      kernel_initializer=tf.orthogonal_initializer,
                                      trainable=True, name=my_scope + '_rnn_in')
        self.convFlat = tf.reshape(self.rnn_in, [self.batch_size, self.trainLength, self.rnn_dim])

        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32,
                                                     initial_state=self.state_in, scope=my_scope + '_rnn',)
        self.rnn = tf.reshape(self.rnn, shape=[-1, self.rnn_dim])
        self.rnn_output = self.rnn

        if extra_layer:
            self.rnn_in2 = tf.layers.dense(self.rnn_output, self.rnn_dim, activation=tf.nn.relu,
                                           kernel_initializer=tf.orthogonal_initializer,
                                           trainable=True, name=my_scope + "_rnn_in_2")
            self.rnnFlat = tf.reshape(self.rnn_in2, [self.batch_size, self.trainLength, self.rnn_dim])

            self.rnn2, self.rnn_state2 = tf.nn.dynamic_rnn(inputs=self.rnnFlat, cell=rnn_cell, dtype=tf.float32,
                                                           initial_state=self.state_in, scope=my_scope + '_rnn2',
                                                           name=my_scope + "_rnn2")
            self.rnn2 = tf.reshape(self.rnn2, shape=[-1, self.rnn_dim])
            self.rnn2_output = self.rnn2
            # The output from the recurrent player is then split into separate Value and Advantage streams
            self.streamA, self.streamV = tf.split(self.rnn2_output, 2, 1)

        else:
            self.rnn_state2 = self.rnn_state
            self.streamA, self.streamV = tf.split(self.rnn_output, 2, 1)
        self.AW = tf.Variable(tf.random_normal([self.rnn_output_size // 2, num_actions]), name=my_scope+"aw")
        self.VW = tf.Variable(tf.random_normal([self.rnn_output_size // 2, 1]), name=my_scope+"vw")
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        #                ------------ Reflected network ------------                   #

        self.ref_left_eye = tf.reverse(self.left_eye, [1])
        self.ref_right_eye = tf.reverse(self.right_eye, [1])

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

        self.conv_with_states_ref = tf.concat(
            [self.conv4l_flat_ref, self.conv4r_flat_ref, tf.reverse(self.prev_actions_one_hot, [1]),
             tf.reverse(self.internal_state, [1])], 1)
        self.rnn_in_ref = tf.layers.dense(self.conv_with_states_ref, self.rnn_dim, activation=tf.nn.relu,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          trainable=True, name=my_scope + '_rnn_in', reuse=True)
        self.convFlat_ref = tf.reshape(self.rnn_in_ref, [self.batch_size, self.trainLength, self.rnn_dim])

        self.rnn_ref, self.rnn_state_ref = tf.nn.dynamic_rnn(inputs=self.convFlat_ref, cell=rnn_cell, dtype=tf.float32,
                                                             initial_state=self.state_in, scope=my_scope + '_rnn')
        self.rnn_ref = tf.reshape(self.rnn_ref, shape=[-1, self.rnn_dim])
        self.rnn_output_ref = self.rnn_ref

        if extra_layer:
            self.rnn_in2_ref = tf.layers.dense(self.rnn_output_ref, self.rnn_dim, activation=tf.nn.relu,
                                               kernel_initializer=tf.orthogonal_initializer,
                                               trainable=True, name=my_scope + "_rnn_in_2", reuse=True)
            self.rnnFlat_ref = tf.reshape(self.rnn_in2_ref, [self.batch_size, self.trainLength, self.rnn_dim])

            self.rnn2_ref, self.rnn_state2_ref = tf.nn.dynamic_rnn(inputs=self.rnnFlat_ref, cell=rnn_cell,
                                                                   dtype=tf.float32,
                                                                   initial_state=self.state_in,
                                                                   scope=my_scope + '_rnn2', name=my_scope + "_rnn2",
                                                                   reuse=True)
            self.rnn2_ref = tf.reshape(self.rnn2_ref, shape=[-1, self.rnn_dim])
            self.rnn2_output_ref = self.rnn2_ref
            # The output from the recurrent player is then split into separate Value and Advantage streams
            self.streamA_ref, self.streamV_ref = tf.split(self.rnn2_output_ref, 2, 1)

        else:
            self.rnn_state2_ref = self.rnn_state_ref
            self.streamA_ref, self.streamV_ref = tf.split(self.rnn_output_ref, 2, 1)

        self.AW_ref = tf.Variable(tf.random_normal([self.rnn_output_size // 2, num_actions]))
        self.VW_ref = tf.Variable(tf.random_normal([self.rnn_output_size // 2, 1]))
        self.Advantage_ref = tf.matmul(self.streamA_ref, self.AW_ref)
        self.Value_ref = tf.matmul(self.streamV_ref, self.VW_ref)

        #                ------------ Integrating Normal and Reflected ------------                   #
        self.Value_final = (self.Value + self.Value_ref)/2

        # TODO: Note that this is specific to the current action space and order (no easy way to make more general
        self.Advantage_final = []
        self.Advantage_final.append((self.Advantage[:, 0] + self.Advantage_ref[:, 0])/2)
        self.Advantage_final.append((self.Advantage[:, 1] + self.Advantage_ref[:, 2])/2)
        self.Advantage_final.append((self.Advantage[:, 2] + self.Advantage_ref[:, 1])/2)
        self.Advantage_final.append((self.Advantage[:, 3] + self.Advantage_ref[:, 3])/2)
        self.Advantage_final.append((self.Advantage[:, 4] + self.Advantage_ref[:, 5])/2)
        self.Advantage_final.append((self.Advantage[:, 5] + self.Advantage_ref[:, 4])/2)
        self.Advantage_final.append((self.Advantage[:, 6] + self.Advantage_ref[:, 6])/2)
        self.Advantage_final.append((self.Advantage[:, 7] + self.Advantage_ref[:, 8])/2)
        self.Advantage_final.append((self.Advantage[:, 8] + self.Advantage_ref[:, 7])/2)
        self.Advantage_final.append((self.Advantage[:, 9] + self.Advantage_ref[:, 9])/2)
        self.Advantage_final = tf.stack(self.Advantage_final)
        self.Advantage_final = tf.reshape(self.Advantage_final, shape=[-1, self.Advantage_final.shape[0]])

        self.salience = tf.gradients(self.Advantage_final, self.observation)
        # Then combine them together to get our final Q-values.
        self.Q_out = self.Value_final + tf.subtract(self.Advantage_final, tf.reduce_mean(self.Advantage_final, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Q_out, 1)
        self.Q_dist = tf.nn.softmax(self.Q_out / self.Temp)
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_one_hot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)

        # In order to only propagate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])
        self.maskB = tf.ones([self.batch_size, self.trainLength // 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)
