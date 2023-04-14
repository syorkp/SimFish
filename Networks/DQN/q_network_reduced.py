# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class QNetworkReduced:

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, num_actions, internal_states=2, learning_rate=0.0001,
                 full_efference_copy=True):
        """The network receives the observation from both eyes, processes it
        #through convolutional layers, concatenates it with the internal state
        #and feeds it to the RNN."""

        self.num_arms = simulation.fish.left_eye.observation_size  # Rays for each eye
        self.rnn_dim = rnn_dim
        self.rnn_output_size = self.rnn_dim
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.actions_one_hot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        if full_efference_copy:
            self.prev_actions = tf.placeholder(shape=[None, 3], dtype=tf.float32, name='prev_actions')
            self.prev_action_consequences = self.prev_actions[:, 1:]
            self.prev_action_impulse = self.prev_action_consequences[:, :1]
            self.prev_action_angle = self.prev_action_consequences[:, 1:]
            self.prev_chosen_actions = self.prev_actions[:, 0]
            self.prev_chosen_actions = tf.cast(self.prev_chosen_actions, dtype=tf.int32)
            self.prev_actions_one_hot = tf.one_hot(self.prev_chosen_actions, num_actions, dtype=tf.float32)
        else:
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
            self.prev_actions_one_hot = tf.one_hot(self.prev_actions, num_actions, dtype=tf.float32)

        self.internal_state = tf.placeholder(shape=[None, internal_states], dtype=tf.float32, name='internal_state')

        #                ------------ Common to Both ------------                   #

        self.exp_keep = tf.placeholder(shape=None, dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None, dtype=tf.float32)
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.rnn_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn_state_in_ref = rnn_cell.zero_state(self.batch_size, tf.float32)

        #                ------------ Normal network ------------                   #

        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levels.

        self.conv_with_states = tf.concat(
            [self.prev_actions_one_hot, self.prev_action_impulse,
             self.prev_action_angle, self.internal_state], 1)

        self.rnn_in = tf.layers.dense(self.conv_with_states, self.rnn_dim, activation=tf.nn.relu,
                                      kernel_initializer=tf.orthogonal_initializer,
                                      trainable=True, name=my_scope + '_rnn_in')
        self.convFlat = tf.reshape(self.rnn_in, [self.batch_size, self.train_length, self.rnn_dim])

        self.rnn, self.rnn_state_shared = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32,
                                                            initial_state=self.rnn_state_in, scope=my_scope + '_rnn', )
        self.rnn = tf.reshape(self.rnn, shape=[-1, self.rnn_dim])
        self.rnn_output = self.rnn

        self.streamA, self.streamV = tf.split(self.rnn_output, 2, 1)
        self.AW = tf.Variable(tf.random_normal([self.rnn_output_size // 2, num_actions]), name=my_scope+"aw")
        self.VW = tf.Variable(tf.random_normal([self.rnn_output_size // 2, 1]), name=my_scope+"vw")
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        #                ------------ Reflected network ------------                   #
        self.prev_actions_one_hot_rev = tf.concat([self.prev_actions_one_hot[0:, :][:, :1],
                                                   self.prev_actions_one_hot[0:, :][:, 2:3],
                                                   self.prev_actions_one_hot[0:, :][:, 1:2],
                                                   self.prev_actions_one_hot[0:, :][:, 3:4],
                                                   self.prev_actions_one_hot[0:, :][:, 5:6],
                                                   self.prev_actions_one_hot[0:, :][:, 4:5],
                                                   self.prev_actions_one_hot[0:, :][:, 6:7],
                                                   self.prev_actions_one_hot[0:, :][:, 8:9],
                                                   self.prev_actions_one_hot[0:, :][:, 7:8],
                                                   self.prev_actions_one_hot[0:, :][:, 9:10],
                                                   self.prev_actions_one_hot[0:, :][:, 11:12],
                                                   self.prev_actions_one_hot[0:, :][:, 10:11],
                                                   ], axis=1)
        self.prev_action_impulse_rev = self.prev_action_impulse
        self.prev_action_angle_rev = tf.multiply(self.prev_action_angle, -1)
        self.conv_with_states_ref = tf.concat(
            [self.prev_actions_one_hot_rev, self.prev_action_impulse_rev, self.prev_action_angle_rev, self.internal_state], 1)

        self.rnn_in_ref = tf.layers.dense(self.conv_with_states_ref, self.rnn_dim, activation=tf.nn.relu,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          trainable=True, name=my_scope + '_rnn_in', reuse=True)
        self.convFlat_ref = tf.reshape(self.rnn_in_ref, [self.batch_size, self.train_length, self.rnn_dim])

        self.rnn_ref, self.rnn_state_ref = tf.nn.dynamic_rnn(inputs=self.convFlat_ref, cell=rnn_cell, dtype=tf.float32,
                                                             initial_state=self.rnn_state_in_ref, scope=my_scope + '_rnn')
        self.rnn_ref = tf.reshape(self.rnn_ref, shape=[-1, self.rnn_dim])
        self.rnn_output_ref = self.rnn_ref

        self.streamA_ref, self.streamV_ref = tf.split(self.rnn_output_ref, 2, 1)

        self.Value_ref = tf.matmul(self.streamV_ref, self.VW)
        self.Advantage_ref = tf.matmul(self.streamA_ref, self.AW)

        # Swapping rows in advantage - Note that this is specific to the current action space and order
        self.Advantage_ref = tf.concat([self.Advantage_ref[0:, :][:, :1],
                                        self.Advantage_ref[0:, :][:, 2:3],
                                        self.Advantage_ref[0:, :][:, 1:2],
                                        self.Advantage_ref[0:, :][:, 3:4],
                                        self.Advantage_ref[0:, :][:, 5:6],
                                        self.Advantage_ref[0:, :][:, 4:5],
                                        self.Advantage_ref[0:, :][:, 6:7],
                                        self.Advantage_ref[0:, :][:, 8:9],
                                        self.Advantage_ref[0:, :][:, 7:8],
                                        self.Advantage_ref[0:, :][:, 9:10],
                                        self.Advantage_ref[0:, :][:, 11:12],
                                        self.Advantage_ref[0:, :][:, 10:11]], axis=1)

        #                ------------ Integrating Normal and Reflected ------------                   #

        self.Value_final = tf.divide(tf.add(self.Value, self.Value_ref), 2)
        self.Advantage_final = tf.divide(tf.add(self.Advantage, self.Advantage_ref), 2)

        self.salience = tf.gradients(self.Advantage_final, self.observation)
        # Then combine them together to get our final Q-values.
        self.Q_out = self.Value_final + tf.subtract(self.Advantage_final,
                                                    tf.reduce_mean(self.Advantage_final, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Q_out, 1)
        self.Q_dist = tf.nn.softmax(self.Q_out / self.Temp)
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_one_hot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)

        # In order to only propagate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size, self.train_length // 2])
        self.maskB = tf.ones([self.batch_size, self.train_length // 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)
