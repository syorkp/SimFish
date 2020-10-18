import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class QNetwork:

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, num_actions, learning_rate=0.0001):
        """The network receives the observation from both eyes, processes it
        through convolutional layers, concatenates it with the internal state
        and feeds it to the RNN."""
        # TODO: Keep in self: rnn_state, streamA, streamV, observation, internal_state,
        #  prev_actions, state_in, exp_keep, predict, trainLength, batch_size, rnn_dim,
        #  Q_out, updateModel, targetQ, actions,

        # Placeholders that are referenced.
        self.num_arms = len(simulation.left_eye.vis_angles)  # Rays for each eye
        self.rnn_dim = rnn_dim
        self.rnn_output_size = self.rnn_dim
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.actions_one_hot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_actions')
        self.prev_actions_one_hot = tf.one_hot(self.prev_actions, num_actions, dtype=tf.float32)

        self.internal_state = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='internal_state')

        self.observation = tf.placeholder(shape=[None, 3, 2], dtype=tf.float32, name='obs')

        self.exp_keep = tf.placeholder(shape=None, dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None, dtype=tf.float32)
        self.trainLength = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        # Start of data stream.
        self.reshaped_observation = tf.reshape(self.observation, shape=[-1, self.num_arms, 3, 2])

        self.left_eye = self.reshaped_observation[:, :, :, 0]
        self.right_eye = self.reshaped_observation[:, :, :, 1]

        # Build the convolutional layers.
        self.conv_with_states = self.build_convolutional_layers(my_scope)

        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levels.
        self.state_in = None
        self.rnn_output, self.rnn_state = self.build_recurrent_layer(my_scope, rnn_cell)

        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA, self.streamV = tf.split(self.rnn_output, 2, 1)
        self.Advantage, self.Value = self.build_two_streams(num_actions)

        # Salience not currently used.
        # self.salience = tf.gradients(self.Advantage, self.observation)

        # Then combine them together to get our final Q-values.
        self.Q_out = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Q_out, 1)

        # TODO: Not used, find out why.
        # self.Q_dist = tf.nn.softmax(self.Q_out / self.Temp)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.loss = self.build_loss_function()

        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.updateModel = trainer.minimize(self.loss)

    def build_convolutional_layers(self, my_scope):
        conv1l = tf.layers.conv1d(inputs=self.left_eye, filters=16, kernel_size=16, strides=4, padding='valid',
                                  activation=tf.nn.relu, name=my_scope + '_conv1l')
        conv2l = tf.layers.conv1d(inputs=conv1l, filters=8, kernel_size=8, strides=2, padding='valid',
                                  activation=tf.nn.relu, name=my_scope + '_conv2l')
        conv3l = tf.layers.conv1d(inputs=conv2l, filters=8, kernel_size=4, strides=1, padding='valid',
                                  activation=tf.nn.relu, name=my_scope + '_conv3l')
        conv4l = tf.layers.conv1d(inputs=conv3l, filters=64, kernel_size=4, strides=1, padding='valid',
                                  activation=tf.nn.relu, name=my_scope + '_conv4l')

        conv1r = tf.layers.conv1d(inputs=self.right_eye, filters=16, kernel_size=16, strides=4, padding='valid',
                                  activation=tf.nn.relu, name=my_scope + '_conv1r')
        conv2r = tf.layers.conv1d(inputs=conv1r, filters=8, kernel_size=8, strides=2, padding='valid',
                                  activation=tf.nn.relu, name=my_scope + '_conv2r')
        conv3r = tf.layers.conv1d(inputs=conv2r, filters=8, kernel_size=4, strides=1, padding='valid',
                                  activation=tf.nn.relu, name=my_scope + '_conv3r')
        conv4r = tf.layers.conv1d(inputs=conv3r, filters=64, kernel_size=4, strides=1, padding='valid',
                                  activation=tf.nn.relu, name=my_scope + '_conv4r')

        conv4l_flat = tf.layers.flatten(conv4l)
        conv4r_flat = tf.layers.flatten(conv4r)

        return tf.concat(
            [conv4l_flat, conv4r_flat, self.prev_actions_one_hot, self.internal_state], 1)

    def build_recurrent_layer(self, my_scope, rnn_cell):
        # TODO: Make my_scope attribute.
        rnn_in = tf.layers.dense(self.conv_with_states, self.rnn_dim, activation=tf.nn.relu,
                                 kernel_initializer=tf.orthogonal_initializer, trainable=True)
        convFlat = tf.reshape(rnn_in, [self.batch_size, self.trainLength, self.rnn_dim])

        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        rnn, rnn_state = tf.nn.dynamic_rnn(inputs=convFlat, cell=rnn_cell, dtype=tf.float32,
                                           initial_state=self.state_in, scope=my_scope + '_rnn')
        rnn = tf.reshape(rnn, shape=[-1, self.rnn_dim])
        return rnn, rnn_state

    def build_two_streams(self, num_actions):
        AW = tf.Variable(tf.random_normal([self.rnn_output_size // 2, num_actions]))
        VW = tf.Variable(tf.random_normal([self.rnn_output_size // 2, 1]))
        Advantage = tf.matmul(self.streamA, AW)
        Value = tf.matmul(self.streamV, VW)
        return Advantage, Value

    def build_loss_function(self):
        Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_one_hot), axis=1)
        td_error = tf.square(self.targetQ - Q)

        # In order to only propagate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        maskA = tf.zeros([self.batch_size, self.trainLength // 2])
        maskB = tf.ones([self.batch_size, self.trainLength // 2])
        mask = tf.concat([maskA, maskB], 1)
        mask = tf.reshape(mask, [-1])
        loss = tf.reduce_mean(td_error * mask)
        return loss
