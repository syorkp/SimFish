import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from Networks.base_network import BaseNetwork

tf.disable_v2_behavior()


class PPONetworkActorMultivariate(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, max_impulse, max_angle_change,
                 clip_param):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=2)

        #            ----------        Non-Reflected       ---------            #

        self.mu_impulse_stream, self.sigma_impulse_stream, self.mu_angle_stream, self.sigma_angle_stream = tf.split(self.rnn_output, 4, 1)

        # Actor impulse output
        # self.mu_impulse = tf.layers.dense(self.mu_impulse_stream, 1, activation=tf.nn.sigmoid,
        #                                   kernel_initializer=tf.orthogonal_initializer,
        #                                   name=my_scope + '_mu_impulse', trainable=True)
        self.mu_impulse = tf.layers.dense(self.mu_impulse_stream, 1, activation=tf.nn.tanh,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          name=my_scope + '_mu_impulse', trainable=True)

        self.sigma_impulse = tf.layers.dense(self.sigma_impulse_stream, 1,
                                             kernel_initializer=tf.orthogonal_initializer,
                                             name=my_scope + '_sigma_impulse', trainable=True)
        self.sigma_impulse = self.bounded_output(self.sigma_impulse, 0, 1)

        # Actor angle output
        self.mu_angle = tf.layers.dense(self.mu_angle_stream, 1, activation=tf.nn.tanh,
                                        kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_mu_angle',
                                        trainable=True)

        self.sigma_angle = tf.layers.dense(self.sigma_angle_stream, 1,
                                           kernel_initializer=tf.orthogonal_initializer,
                                           name=my_scope + '_sigma_angle', trainable=True)
        self.sigma_angle = self.bounded_output(self.sigma_angle, 0, 1)

        #            ----------        Reflected       ---------            #

        self.mu_impulse_stream_ref, self.sigma_impulse_stream_ref, self.mu_angle_stream_ref, self.sigma_angle_stream_ref = tf.split(self.rnn_output_ref, 4, 1)

        # Actor impulse output
        # self.mu_impulse_ref = tf.layers.dense(self.mu_impulse_stream_ref, 1, activation=tf.nn.sigmoid,
        #                                       kernel_initializer=tf.orthogonal_initializer,
        #                                       name=my_scope + '_mu_impulse', reuse=True, trainable=True)
        self.mu_impulse_ref = tf.layers.dense(self.mu_impulse_stream_ref, 1, activation=tf.nn.tanh,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              name=my_scope + '_mu_impulse', reuse=True, trainable=True)

        self.sigma_impulse_ref = tf.layers.dense(self.sigma_impulse_stream_ref, 1,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 name=my_scope + '_sigma_impulse', reuse=True, trainable=True)
        self.sigma_impulse_ref = self.bounded_output(self.sigma_impulse_ref, 0, 1)

        # Actor angle output
        self.mu_angle_ref = tf.layers.dense(self.mu_angle_stream_ref, 1, activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer,
                                            name=my_scope + '_mu_angle', reuse=True, trainable=True)

        self.sigma_angle_ref = tf.layers.dense(self.sigma_angle_stream_ref, 1,
                                               kernel_initializer=tf.orthogonal_initializer,
                                               name=my_scope + '_sigma_angle', reuse=True, trainable=True)
        self.sigma_angle_ref = self.bounded_output(self.sigma_angle_ref, 0, 1)

        #            ----------        Combined       ---------            #

        # Combined Actor impulse output
        self.mu_impulse_combined = tf.math.divide(tf.math.add(self.mu_impulse, self.mu_impulse_ref), 2.0,
                                                  name="mu_impulse_combined")
        self.sigma_impulse_combined = tf.math.divide(tf.math.add(self.sigma_impulse, self.sigma_impulse_ref), 2.0,
                                                     name="sigma_impulse_combined")

        # Combined Actor angle output
        self.mu_angle_combined = tf.math.divide(tf.math.subtract(self.mu_angle, self.mu_angle_ref), 2.0,
                                                name="mu_angle_combined")
        self.sigma_angle_combined = tf.math.divide(tf.math.add(self.sigma_angle, self.sigma_angle_ref), 2.0,
                                                   name="sigma_angle_combined")

        # Multinomial distribution
        self.mu_action = tf.concat([self.mu_impulse_combined, self.mu_angle_combined], axis=1)
        self.sigma_action = tf.concat([self.sigma_impulse_combined, self.sigma_angle_combined], axis=1)
        self.action_distribution = tfp.distributions.MultivariateNormalDiag(loc=self.mu_action, scale_diag=self.sigma_action)

        self.action_output = tf.squeeze(self.action_distribution.sample(1), axis=0)

        self.impulse_output, self.angle_output = tf.split(self.action_output, 2, axis=1)
        # Scaled impulse TODO: Remove if doesnt work
        self.impulse_output = tf.divide(tf.add(self.impulse_output, 1), 2)
        self.impulse_output = tf.clip_by_value(self.impulse_output, 0, 1)
        self.angle_output = tf.clip_by_value(self.angle_output, -1, 1)
        self.impulse_output = tf.math.multiply(self.impulse_output, max_impulse, name="impulse_output")
        self.angle_output = tf.math.multiply(self.angle_output, max_angle_change, name="angle_output")

        self.log_prob = tf.log(self.action_distribution.prob(self.action_output) + 1e-5)

        #            ----------        Loss functions       ---------            #

        # Placeholders
        self.action_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='impulse_placeholder')
        self.impulse_placeholder, self.angle_placeholder = tf.split(self.action_placeholder, 2, axis=1)
        self.impulse_placeholder = tf.math.divide(self.impulse_placeholder, max_impulse)
        # Scaled impulse TODO: Remove if doesnt work
        self.impulse_placeholder = tf.math.multiply(tf.math.subtract(self.impulse_placeholder, 1), 2)
        self.angle_placeholder = tf.math.divide(self.angle_placeholder, max_angle_change)
        self.action_placeholder2 = tf.concat([self.impulse_placeholder, self.angle_placeholder], axis=1)

        self.new_log_prob = tf.log(self.action_distribution.prob(self.action_placeholder2) + 1e-5)

        self.old_log_prob = tf.placeholder(shape=[None], dtype=tf.float32, name='old_log_prob_impulse')

        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')

        # COMBINED LOSS

        self.ratio = tf.exp(self.new_log_prob - self.old_log_prob)
        self.surrogate_loss_1 = tf.math.multiply(self.ratio, self.scaled_advantage_placeholder)
        self.surrogate_loss_2 = tf.math.multiply(
            tf.clip_by_value(self.ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)
        self.total_loss = -tf.reduce_mean(tf.minimum(self.surrogate_loss_1, self.surrogate_loss_2))

        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='actor_optimizer_impulse').minimize(self.total_loss)

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower

