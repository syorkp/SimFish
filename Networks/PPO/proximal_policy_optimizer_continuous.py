import tensorflow.compat.v1 as tf

from Networks.base_network import BaseNetwork

tf.disable_v2_behavior()


class PPONetworkActor(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, max_impulse, max_angle_change,
                 clip_param):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=2)

        #            ----------        Non-Reflected       ---------            #

        self.mu_impulse_stream, self.mu_angle_stream = tf.split(self.rnn_output, 2, 1)

        # Actor impulse output
        self.mu_impulse = tf.layers.dense(self.mu_impulse_stream, 1, activation=tf.nn.sigmoid,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          name=my_scope + '_mu_impulse', trainable=True)

        # self.sigma_impulse = tf.layers.dense(self.sigma_impulse_stream, 1,
        #                                      kernel_initializer=tf.orthogonal_initializer,
        #                                      name=my_scope + '_sigma_impulse', trainable=True)

        # Actor angle output
        self.mu_angle = tf.layers.dense(self.mu_angle_stream, 1, activation=tf.nn.tanh,
                                        kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_mu_angle',
                                        trainable=True)

        # self.sigma_angle = tf.layers.dense(self.sigma_angle_stream, 1,
        #                                    kernel_initializer=tf.orthogonal_initializer,
        #                                    name=my_scope + '_sigma_angle', trainable=True)

        #            ----------        Reflected       ---------            #

        self.mu_impulse_stream_ref, self.mu_angle_stream_ref = tf.split(self.rnn_output_ref, 2, 1)

        # Actor impulse output
        self.mu_impulse_ref = tf.layers.dense(self.mu_impulse_stream_ref, 1, activation=tf.nn.sigmoid,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              name=my_scope + '_mu_impulse', reuse=True, trainable=True)

        # self.sigma_impulse_ref = tf.layers.dense(self.sigma_impulse_stream_ref, 1,
        #                                          kernel_initializer=tf.orthogonal_initializer,
        #                                          name=my_scope + '_sigma_impulse', reuse=True, trainable=True)

        # Actor angle output
        self.mu_angle_ref = tf.layers.dense(self.mu_angle_stream_ref, 1, activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer,
                                            name=my_scope + '_mu_angle', reuse=True, trainable=True)

        # self.sigma_angle_ref = tf.layers.dense(self.sigma_angle_stream_ref, 1,
        #                                        kernel_initializer=tf.orthogonal_initializer,
        #                                        name=my_scope + '_sigma_angle', reuse=True, trainable=True)

        #            ----------        Combined       ---------            #

        # Combined Actor impulse output
        self.mu_impulse_combined = tf.math.divide(tf.math.add(self.mu_impulse, self.mu_impulse_ref), 2.0,
                                                  name="mu_impulse_combined")
        # self.sigma_impulse_combined = tf.math.divide(tf.math.add(self.sigma_impulse, self.sigma_impulse_ref), 2.0,
        #                                              name="sigma_impulse_combined")
        self.sigma_impulse_combined = tf.placeholder(shape=[None], dtype=tf.float32, name='sigma_impulse_combined')
        self.norm_dist_impulse = tf.distributions.Normal(self.mu_impulse_combined, self.sigma_impulse_combined,
                                                         name="norm_dist_impulse")
        self.action_tf_var_impulse = tf.squeeze(self.norm_dist_impulse.sample(1), axis=0)
        self.action_tf_var_impulse = tf.clip_by_value(self.action_tf_var_impulse, 0, 1)
        self.impulse_output = tf.math.multiply(self.action_tf_var_impulse, max_impulse, name="impulse_output")
        self.log_prob_impulse = tf.log(self.norm_dist_impulse.prob(self.action_tf_var_impulse) + 1e-5)

        # Combined Actor angle output
        self.mu_angle_combined = tf.math.divide(tf.math.subtract(self.mu_angle, self.mu_angle_ref), 2.0,
                                                name="mu_angle_combined")
        # self.sigma_angle_combined = tf.math.divide(tf.math.add(self.sigma_angle, self.sigma_angle_ref), 2.0,
        #                                            name="sigma_angle_combined")
        self.sigma_angle_combined = tf.placeholder(shape=[None], dtype=tf.float32, name='sigma_angle_combined')
        self.norm_dist_angle = tf.distributions.Normal(self.mu_angle_combined, self.sigma_angle_combined,
                                                       name="norm_dist_angle")
        self.action_tf_var_angle = tf.squeeze(self.norm_dist_angle.sample(1), axis=0)
        self.action_tf_var_angle = tf.clip_by_value(self.action_tf_var_angle, -1, 1)
        self.angle_output = tf.math.multiply(self.action_tf_var_angle, max_angle_change, name="angle_output")
        self.log_prob_angle = tf.log(self.norm_dist_angle.prob(self.action_tf_var_angle) + 1e-5)

        # self.mu_impulse_combined = self.mu_impulse
        # self.sigma_impulse_combined = self.sigma_impulse
        # self.norm_dist_impulse = tf.distributions.Normal(self.mu_impulse_combined, self.sigma_impulse_combined,
        #                                                  name="norm_dist_impulse")
        # self.action_tf_var_impulse = tf.squeeze(self.norm_dist_impulse.sample(1), axis=0)
        # self.action_tf_var_impulse = tf.clip_by_value(self.action_tf_var_impulse, 0, 1)
        # self.impulse_output = tf.math.multiply(self.action_tf_var_impulse, max_impulse, name="impulse_output")
        # self.log_prob_impulse = tf.log(self.norm_dist_impulse.prob(self.action_tf_var_impulse) + 1e-5)
        #
        # # Combined Actor angle output
        # self.mu_angle_combined = self.mu_angle
        # self.sigma_angle_combined = self.sigma_angle
        # self.norm_dist_angle = tf.distributions.Normal(self.mu_angle_combined, self.sigma_angle_combined,
        #                                                name="norm_dist_angle")
        # self.action_tf_var_angle = tf.squeeze(self.norm_dist_angle.sample(1), axis=0)
        # self.action_tf_var_angle = tf.clip_by_value(self.action_tf_var_angle, -1, 1)
        # self.angle_output = tf.math.multiply(self.action_tf_var_angle, max_angle_change, name="angle_output")
        # self.log_prob_angle = tf.log(self.norm_dist_angle.prob(self.action_tf_var_angle) + 1e-5)

        #            ----------        Loss functions       ---------            #

        # Placeholders
        self.impulse_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='impulse_placeholder')
        self.angle_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='angle_placeholder')

        self.new_log_prob_impulse = tf.log(self.norm_dist_impulse.prob(tf.math.divide(self.impulse_placeholder, max_impulse)) + 1e-5)
        self.new_log_prob_angle = tf.log(self.norm_dist_angle.prob(tf.math.divide(self.angle_placeholder, max_angle_change)) + 1e-5)

        self.old_log_prob_impulse_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='old_log_prob_impulse')
        self.old_log_prob_angle_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='old_log_prob_angle')

        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')

        # COMBINED LOSS

        self.impulse_ratio = tf.exp(self.new_log_prob_impulse - self.old_log_prob_impulse_placeholder)
        self.impulse_surrogate_loss_1 = tf.math.multiply(self.impulse_ratio, self.scaled_advantage_placeholder)
        self.impulse_surrogate_loss_2 = tf.math.multiply(
            tf.clip_by_value(self.impulse_ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)
        self.impulse_loss = -tf.reduce_mean(tf.minimum(self.impulse_surrogate_loss_1, self.impulse_surrogate_loss_2))

        self.angle_ratio = tf.exp(self.new_log_prob_angle - self.old_log_prob_angle_placeholder)
        self.angle_surrogate_loss_1 = tf.math.multiply(self.angle_ratio, self.scaled_advantage_placeholder)
        self.angle_surrogate_loss_2 = tf.math.multiply(
            tf.clip_by_value(self.angle_ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)
        self.angle_loss = -tf.reduce_mean(tf.minimum(self.angle_surrogate_loss_1, self.angle_surrogate_loss_2))

        self.total_loss = tf.add(self.impulse_loss, self.angle_loss)

        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='actor_optimizer_impulse').minimize(self.total_loss)

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower

