import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class A2CNetwork:

    def __init__(self, simulation, rnn_dim_shared, rnn_dim_critic, rnn_dim_actor, rnn_cell_shared, rnn_cell_critic,
                 rnn_cell_actor, my_scope, internal_states=2, actor_learning_rate_impulse=0.00001,
                 actor_learning_rate_angle=0.00001,
                 critic_learning_rate=0.00056, max_impulse=80.0, max_angle_change=1.0, gradient_clip=1.0):
        # Variables
        self.num_arms = len(simulation.fish.left_eye.vis_angles)  # Rays for each eye
        self.rnn_dim_shared = rnn_dim_shared
        self.rnn_dim_critic = rnn_dim_critic
        self.rnn_dim_actor = rnn_dim_actor

        self.rnn_output_size_shared = self.rnn_dim_shared

        self.trainLength = tf.placeholder(dtype=tf.int32, name="train_length")
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')

        self.shared_state_in = rnn_cell_shared.zero_state(self.batch_size, tf.float32)
        self.critic_state_in = rnn_cell_critic.zero_state(self.batch_size, tf.float32)
        self.actor_state_in = rnn_cell_actor.zero_state(self.batch_size, tf.float32)

        # Network Inputs
        self.prev_actions = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='prev_actions')
        self.internal_state = tf.placeholder(shape=[None, internal_states], dtype=tf.float32, name='internal_state')

        self.observation = tf.placeholder(shape=[None, 3, 2], dtype=tf.float32, name='obs')

        self.observation = tf.placeholder(shape=[None, 3, 2], dtype=tf.float32, name='obs')
        self.scaler = tf.placeholder(shape=[None, 3, 2], dtype=tf.float32, name='scaler')
        self.scaled_obs = tf.divide(self.observation, self.scaler, name="scaled_observation")

        self.reshaped_observation = tf.reshape(self.scaled_obs, shape=[-1, self.num_arms, 3, 2],
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
        self.rnn_in = tf.layers.dense(self.conv_with_states, self.rnn_dim_shared, activation=tf.nn.relu,
                                      kernel_initializer=tf.orthogonal_initializer,
                                      trainable=True, name=my_scope + '_rnn_in')
        self.convFlat = tf.reshape(self.rnn_in, [self.batch_size, self.trainLength, self.rnn_dim_shared],
                                   name="flattened_shared_rnn_input")

        self.rnn, self.rnn_state_shared = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell_shared,
                                                            dtype=tf.float32,
                                                            initial_state=self.shared_state_in, scope=my_scope + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, self.rnn_dim_shared], name="shared_rnn_output")
        self.rnn_output = self.rnn
        self.rnn_state2 = self.rnn_state_shared

        # Critic (value) output
        # self.value_rnn_in = tf.layers.dense(self.rnn_output, self.rnn_dim_critic, None,
        #                                     kernel_initializer=tf.orthogonal_initializer,
        #                                     name=my_scope + '_rnn_value_in',
        #                                     trainable=True)
        # self.flat_value_rnn_in = tf.reshape(self.value_rnn_in, [self.batch_size, self.trainLength, self.rnn_dim_critic],
        #                                     name="flattened_value_rnn_input")
        # self.value_rnn, self.value_rnn_state = tf.nn.dynamic_rnn(inputs=self.flat_value_rnn_in, cell=rnn_cell_critic,
        #                                                          dtype=tf.float32,
        #                                                          initial_state=self.critic_state_in,
        #                                                          scope=my_scope + '_rnn_value')
        # self.value_rnn = tf.reshape(self.value_rnn, shape=[-1, self.rnn_dim_critic], name="value_rnn_output")
        # self.value_rnn_output = self.value_rnn
        # self.Value = tf.layers.dense(self.value_rnn_output, 1, None,
        #                              kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_Value',
        #                              trainable=True)

        self.Value = tf.layers.dense(self.rnn_output, 1, None,
                                     kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_Value',
                                     trainable=True)

        # # Policy RNN
        # self.actor_rnn_in = tf.layers.dense(self.rnn_output, self.rnn_dim_actor, None,
        #                                     kernel_initializer=tf.orthogonal_initializer,
        #                                     name=my_scope + '_rnn_actor_in', trainable=True)
        # self.flat_actor_rnn_in = tf.reshape(self.actor_rnn_in,
        #                                     [self.batch_size, self.trainLength, self.rnn_dim_actor],
        #                                     name=my_scope + '_rnn_actor_in_reshaped')
        # self.actor_rnn, self.actor_rnn_state = tf.nn.dynamic_rnn(inputs=self.flat_actor_rnn_in, cell=rnn_cell_actor,
        #                                                          dtype=tf.float32,
        #                                                          initial_state=self.actor_state_in,
        #                                                          scope=my_scope + '_rnn_actor')
        # self.actor_rnn = tf.reshape(self.actor_rnn, shape=[-1, self.rnn_dim_actor])
        # self.actor_rnn_output = self.actor_rnn

        # Actor impulse output
        self.mu_impulse = tf.layers.dense(self.rnn_output, 1, activation=tf.nn.sigmoid,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          name=my_scope + '_mu_impulse', trainable=True)

        self.sigma_impulse = tf.layers.dense(self.rnn_output, 1, activation=tf.nn.sigmoid,
                                             kernel_initializer=tf.orthogonal_initializer,
                                             name=my_scope + '_sigma_impulse', trainable=True)

        # Actor angle output
        self.mu_angle = tf.layers.dense(self.rnn_output, 1, activation=tf.nn.tanh,
                                        kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_mu_angle',
                                        trainable=True)

        self.sigma_angle = tf.layers.dense(self.rnn_output, 1, activation=tf.nn.sigmoid,
                                           kernel_initializer=tf.orthogonal_initializer,
                                           name=my_scope + '_sigma_angle', trainable=True)

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
        self.prev_actions_ref = tf.reverse(self.prev_actions, [1])
        self.internal_state_ref = tf.reverse(self.internal_state, [1])

        self.conv_with_states_ref = tf.concat(
            [self.conv4l_flat_ref, self.conv4r_flat_ref, self.prev_actions_ref, self.internal_state_ref], 1)
        self.rnn_in_ref = tf.layers.dense(self.conv_with_states_ref, self.rnn_dim_shared, activation=tf.nn.relu,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          trainable=True, name=my_scope + '_rnn_in', reuse=True)
        self.convFlat_ref = tf.reshape(self.rnn_in_ref, [self.batch_size, self.trainLength, self.rnn_dim_shared])

        self.rnn_ref, self.rnn_state_ref = tf.nn.dynamic_rnn(inputs=self.convFlat_ref, cell=rnn_cell_shared,
                                                             dtype=tf.float32,
                                                             initial_state=self.shared_state_in,
                                                             scope=my_scope + '_rnn_ref')  # No need to reuse as takes rnn_cell as argument for both.
        self.rnn_ref = tf.reshape(self.rnn_ref, shape=[-1, self.rnn_dim_shared])
        self.rnn_output_ref = self.rnn_ref

        # Critic RNN
        # self.value_rnn_in_ref = tf.layers.dense(self.rnn_output_ref, self.rnn_dim_critic, None,
        #                                         kernel_initializer=tf.orthogonal_initializer,
        #                                         name=my_scope + '_rnn_value_in',
        #                                         trainable=True, reuse=True)
        # self.flat_value_rnn_in_ref = tf.reshape(self.value_rnn_in_ref,
        #                                         [self.batch_size, self.trainLength, self.rnn_dim_critic])
        # self.value_rnn_ref, self.value_rnn_state_ref = tf.nn.dynamic_rnn(inputs=self.flat_value_rnn_in_ref,
        #                                                                  cell=rnn_cell_critic,
        #                                                                  dtype=tf.float32,
        #                                                                  initial_state=self.critic_state_in,
        #                                                                  scope=my_scope + '_rnn_value')
        # self.value_rnn_ref = tf.reshape(self.value_rnn_ref, shape=[-1, self.rnn_dim_critic])
        # self.value_rnn_output_ref = self.value_rnn_ref
        # self.Value_ref = tf.layers.dense(self.value_rnn_output_ref, 1, None,
        #                                  kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_Value',
        #                                  reuse=True, trainable=True)
        self.Value_ref = tf.layers.dense(self.rnn_output_ref, 1, None,
                                         kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_Value',
                                         reuse=True, trainable=True)
        #
        # # Policy RNN
        # self.actor_rnn_in_ref = tf.layers.dense(self.rnn_output_ref, self.rnn_dim_actor, None,
        #                                         kernel_initializer=tf.orthogonal_initializer,
        #                                         name=my_scope + '_rnn_actor_in', reuse=True,
        #                                         trainable=True)
        # self.flat_actor_rnn_in_ref = tf.reshape(self.actor_rnn_in_ref,
        #                                         [self.batch_size, self.trainLength, self.rnn_dim_actor],
        #                                         name=my_scope + '_rnn_actor_in_reshaped')
        # self.actor_rnn_ref, self.actor_rnn_state_ref = tf.nn.dynamic_rnn(inputs=self.flat_actor_rnn_in_ref,
        #                                                                  cell=rnn_cell_actor,
        #                                                                  dtype=tf.float32,
        #                                                                  initial_state=self.actor_state_in,
        #                                                                  scope=my_scope + '_rnn_actor')
        # self.actor_rnn_ref = tf.reshape(self.actor_rnn_ref, shape=[-1, self.rnn_dim_actor])
        # self.actor_rnn_output_ref = self.actor_rnn_ref

        # Actor impulse output
        self.mu_impulse_ref = tf.layers.dense(self.rnn_output_ref, 1, activation=tf.nn.sigmoid,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              name=my_scope + '_mu_impulse', reuse=True, trainable=True)
        # self.mu_impulse_ref = tf.math.abs(self.mu_impulse)

        self.sigma_impulse_ref = tf.layers.dense(self.rnn_output_ref, 1, activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 name=my_scope + '_sigma_impulse', reuse=True, trainable=True)
        # self.sigma_impulse_ref = tf.math.abs(self.sigma_impulse_ref)

        # Actor angle output
        self.mu_angle_ref = tf.layers.dense(self.rnn_output_ref, 1, activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer,
                                            name=my_scope + '_mu_angle', reuse=True, trainable=True)

        self.sigma_angle_ref = tf.layers.dense(self.rnn_output_ref, 1, activation=tf.nn.sigmoid,
                                               kernel_initializer=tf.orthogonal_initializer,
                                               name=my_scope + '_sigma_angle', reuse=True, trainable=True)
        # self.sigma_angle_ref = tf.math.abs(self.sigma_angle_ref)


        #            ----------        Combined       ---------            #

        # Combined Actor impulse output
        self.mu_impulse_combined = tf.math.divide(tf.math.add(self.mu_impulse, self.mu_impulse_ref), 2.0,
                                                  name="mu_impulse_combined")
        self.sigma_impulse_combined = tf.math.divide(tf.math.add(self.sigma_impulse, self.sigma_impulse_ref), 2.0,
                                                     name="sigma_impulse_combined")
        self.norm_dist_impulse = tf.distributions.Normal(self.mu_impulse_combined, self.sigma_impulse_combined,
                                                         name="norm_dist_impulse")
        self.action_tf_var_impulse = tf.squeeze(self.norm_dist_impulse.sample(1), axis=0)
        # self.action_tf_var_impulse = tf.math.abs(self.action_tf_var_impulse)  TODO: Trying without
        self.action_tf_var_impulse = tf.clip_by_value(self.action_tf_var_impulse, 0, 1)
        self.impulse_output = tf.math.multiply(self.action_tf_var_impulse, max_impulse, name="impulse_output")

        # Combined Actor angle output
        self.mu_angle_combined = tf.math.divide(tf.math.subtract(self.mu_angle, self.mu_angle_ref), 2.0,
                                                name="mu_angle_combined")
        self.sigma_angle_combined = tf.math.divide(tf.math.add(self.sigma_angle, self.sigma_angle_ref), 2.0,
                                                   name="sigma_angle_combined")
        self.norm_dist_angle = tf.distributions.Normal(self.mu_angle_combined, self.sigma_angle_combined,
                                                       name="norm_dist_angle")
        self.action_tf_var_angle = tf.squeeze(self.norm_dist_angle.sample(1), axis=0)
        self.action_tf_var_angle = tf.clip_by_value(self.action_tf_var_angle, -1, 1)
        self.angle_output = tf.math.multiply(self.action_tf_var_angle, max_angle_change, name="angle_output")

        self.Value_output = tf.math.divide(tf.math.add(self.Value, self.Value_ref), 2, name="value_output")

        #            ----------        Loss functions       ---------            #

        self.action_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='action_placeholder')
        self.delta_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='delta')
        self.target_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='target')

        # Actor (policy) loss function - Impulse
        self.loss_actor_impulse = -tf.log(self.norm_dist_impulse.prob(
            tf.math.divide(self.action_placeholder[0][0], max_impulse)) + 1e-5) * self.delta_placeholder
        self.training_op_actor_impulse = tf.train.AdamOptimizer(actor_learning_rate_impulse,
                                                                name='actor_optimizer_impulse').minimize(self.loss_actor_impulse)

        # Actor (policy) loss function - Angle
        self.loss_actor_angle = -tf.log(self.norm_dist_angle.prob(
            tf.math.divide(self.action_placeholder[0][1], max_angle_change)) + 1e-5) * self.delta_placeholder
        self.training_op_actor_angle = tf.train.AdamOptimizer(actor_learning_rate_angle,
                                                              name='actor_optimizer_angle').minimize(self.loss_actor_angle)

        self.action_op = tf.group(self.training_op_actor_impulse, self.training_op_actor_angle)

        # Critic (state-value) loss function
        self.loss_critic = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.Value_output), self.target_placeholder))
        self.training_op_critic = tf.train.AdamOptimizer(critic_learning_rate, name='critic_optimizer').minimize(
            self.loss_critic)

