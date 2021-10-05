import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from Networks.base_network import BaseNetwork
from Networks.Distributions.reflected_continuous import ReflectedProbabilityDist

tf.disable_v2_behavior()


class PPONetworkActorMultivariate2(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, max_impulse, max_angle_change,
                 clip_param):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=2)

        self._pdtype = ReflectedProbabilityDist(2) # TODO: check correct shape

        #            ----------        Non-Reflected       ---------            #

        self.action_stream, self.value_stream = tf.split(self.rnn_output, 2, 1)


        # self.mu_impulse_stream, self.mu_angle_stream = tf.split(self.rnn_output, 2, 1)

        # # Actor impulse output
        # self.mu_impulse = tf.layers.dense(self.mu_impulse_stream, 1, activation=tf.nn.sigmoid,
        #                                   kernel_initializer=tf.orthogonal_initializer,
        #                                   name=my_scope + '_mu_impulse', trainable=True)
        #
        # # Actor angle output
        # self.mu_angle = tf.layers.dense(self.mu_angle_stream, 1, activation=tf.nn.tanh,
        #                                 kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_mu_angle',
        #                                 trainable=True)

        #            ----------        Reflected       ---------            #

        self.action_stream_ref, self.value_stream_ref = tf.split(self.rnn_output_ref, 2, 1)

        # self.mu_impulse_stream_ref, self.mu_angle_stream_ref = tf.split(self.rnn_output_ref, 2, 1)

        # Actor impulse output
        # self.mu_impulse_ref = tf.layers.dense(self.mu_impulse_stream_ref, 1, activation=tf.nn.sigmoid,
        #                                       kernel_initializer=tf.orthogonal_initializer,
        #                                       name=my_scope + '_mu_impulse', reuse=True, trainable=True)
        #
        # # Actor angle output
        # self.mu_angle_ref = tf.layers.dense(self.mu_angle_stream_ref, 1, activation=tf.nn.tanh,
        #                                     kernel_initializer=tf.orthogonal_initializer,
        #                                     name=my_scope + '_mu_angle', reuse=True, trainable=True)

        #            ----------        Combined       ---------            #

        # Combined Actor impulse output
        # self.mu_impulse_combined = tf.math.divide(tf.math.add(self.mu_impulse, self.mu_impulse_ref), 2.0,
        #                                           name="mu_impulse_combined")
        #
        # # Combined Actor angle output
        # self.mu_angle_combined = tf.math.divide(tf.math.subtract(self.mu_angle, self.mu_angle_ref), 2.0,
        #                                         name="mu_angle_combined")
        #
        # self.sigma_action = tf.get_variable(name='sigmas', shape=[1, 2], initializer=tf.zeros_initializer())
        # self.sigma_action = self.bounded_output(self.sigma_action, 0, 1)

        self.action_distribution, self.mu_action, self.sigma_action, self.q_value = \
            self.pdtype.proba_distribution_from_latent(self.action_stream, self.action_stream_ref, self.value_stream,
                                                       self.value_stream_ref, init_scale=0.01)

        self.mu_impulse_combined, self.mu_angle_combined = tf.split(self.mu_action, 2, axis=1)

        self.action_output = self.action_distribution.sample()
        self.impulse_output, self.angle_output = tf.split(self.action_output, 2, axis=1)

        self.impulse_output = tf.clip_by_value(self.impulse_output, 0, 1)
        self.angle_output = tf.clip_by_value(self.angle_output, -1, 1)

        self.impulse_output = tf.math.multiply(self.impulse_output, max_impulse, name="impulse_output")
        self.angle_output = tf.math.multiply(self.angle_output, max_angle_change, name="angle_output")

        self.log_prob = self.action_distribution.logp(self.action_output)

        #            ----------        Value Outputs       ----------           #

        self.value_fn_1 = tf.layers.dense(self.value_stream, 1, name='vf')
        self.value_fn_2 = tf.layers.dense(self.value_stream_ref, 1, name='vf', reuse=True)

        self.value_output = tf.math.divide(tf.math.add(self.value_fn_1, self.value_fn_2), 2)

        #            ----------        Loss functions       ---------            #

        # Actor loss
        self.action_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='action_placeholder')
        self.impulse_placeholder, self.angle_placeholder = tf.split(self.action_placeholder, 2, axis=1)
        self.impulse_placeholder = tf.math.divide(self.impulse_placeholder, max_impulse)

        self.angle_placeholder = tf.math.divide(self.angle_placeholder, max_angle_change)
        self.action_placeholder2 = tf.concat([self.impulse_placeholder, self.angle_placeholder], axis=1)

        self.new_log_prob = self.action_distribution.logp(self.action_placeholder2)

        self.old_log_prob = tf.placeholder(shape=[None], dtype=tf.float32, name='old_log_prob_impulse')

        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')

        self.ratio = tf.exp(self.old_log_prob - self.new_log_prob)  # TODO: Check that log probs are negative.
        self.surrogate_loss_1 = -tf.math.multiply(self.ratio, self.scaled_advantage_placeholder)
        self.surrogate_loss_2 = -tf.math.multiply(
            tf.clip_by_value(self.ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)
        self.policy_loss = -tf.reduce_mean(tf.maximum(self.surrogate_loss_1, self.surrogate_loss_2))

        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        # Value loss
        self.returns_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='returns')
        self.old_value_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value')
        self.value_cliprange_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='value_cliprange')
        self.value_clipped = self.old_value_placeholder + tf.clip_by_value(self.value_output-self.old_value_placeholder, -self.value_cliprange_placeholder, self.value_cliprange_placeholder)

        self.critic_loss_1 = tf.squared_difference(tf.squeeze(self.value_output), self.returns_placeholder)
        self.critic_loss_2 = tf.squared_difference(tf.squeeze(self.value_clipped), self.returns_placeholder)
        self.value_loss = .5 * tf.reduce_mean(tf.maximum(self.critic_loss_1, self.critic_loss_2))
        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        # Entropy
        self.entropy = tf.reduce_mean(self.action_distribution.entropy())

        # Combined loss
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.total_loss = self.policy_loss - tf.multiply(self.entropy, self.entropy_coefficient) + \
                          tf.multiply(self.value_loss, self.value_coefficient)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='optimizer').minimize(
            self.total_loss)



    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype
