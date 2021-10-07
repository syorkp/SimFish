import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from Networks.base_network import BaseNetwork
from Networks.Distributions.reflected_continuous import ReflectedProbabilityDist

tf.disable_v2_behavior()


class PPONetworkActorMultivariate2(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, max_impulse, max_angle_change,
                 clip_param):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=2)

        self._pdtype = ReflectedProbabilityDist(2)

        #            ----------        Non-Reflected       ---------            #

        self.action_stream, self.value_stream = tf.split(self.rnn_output, 2, 1)

        #            ----------        Reflected       ---------            #

        self.action_stream_ref, self.value_stream_ref = tf.split(self.rnn_output_ref, 2, 1)

        #            ----------        Combined       ---------            #

        self.action_distribution, self.mu_action, self.sigma_action, self.q_value = \
            self.pdtype.proba_distribution_from_latent(self.action_stream, self.action_stream_ref, self.value_stream,
                                                       self.value_stream_ref)

        self.mu_impulse_combined, self.mu_angle_combined = tf.split(self.mu_action, 2, axis=1)

        self.action_output = self.action_distribution.sample()
        self.impulse_output, self.angle_output = tf.split(self.action_output, 2, axis=1)

        self.impulse_output = tf.clip_by_value(self.impulse_output, 0, 1)
        self.angle_output = tf.clip_by_value(self.angle_output, -1, 1)

        self.impulse_output = tf.math.multiply(self.impulse_output, max_impulse, name="impulse_output")
        self.angle_output = tf.math.multiply(self.angle_output, max_angle_change, name="angle_output")

        self.neg_log_prob = self.action_distribution.neglogp(self.action_output)

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
        self.normalised_action = tf.concat([self.impulse_placeholder, self.angle_placeholder], axis=1)

        self.new_neg_log_prob = self.action_distribution.neglogp(self.normalised_action)
        self.old_neg_log_prob = tf.placeholder(shape=[None], dtype=tf.float32, name='old_log_prob_impulse')
        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')

        self.ratio = tf.exp(self.new_neg_log_prob - self.old_neg_log_prob)
        self.surrogate_loss_1 = -tf.math.multiply(self.ratio, self.scaled_advantage_placeholder)
        self.surrogate_loss_2 = -tf.math.multiply(
            tf.clip_by_value(self.ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)
        self.policy_loss = tf.reduce_mean(tf.maximum(self.surrogate_loss_1, self.surrogate_loss_2))

        # Value loss
        self.returns_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='returns')
        self.old_value_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value')
        # self.value_cliprange_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='value_cliprange')
        # Clip the different between old and new value NOTE: this depends on the reward scaling
        self.value_clipped = self.old_value_placeholder + tf.clip_by_value(self.value_output-self.old_value_placeholder, -clip_param, clip_param)

        self.critic_loss_1 = tf.squared_difference(tf.squeeze(self.value_output), self.returns_placeholder)
        self.critic_loss_2 = tf.squared_difference(tf.squeeze(self.value_clipped), self.returns_placeholder)
        self.value_loss = .5 * tf.reduce_mean(tf.maximum(self.critic_loss_1, self.critic_loss_2))

        # Entropy
        self.entropy = tf.reduce_mean(self.action_distribution.entropy())

        # Combined loss
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5

        self.total_loss = self.policy_loss - tf.multiply(self.entropy, self.entropy_coefficient) + \
                          tf.multiply(self.value_loss, self.value_coefficient)
        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        # Gradient clipping (for stability)
        self.model_params = tf.trainable_variables()
        self.gradients = tf.gradients(self.total_loss, self.model_params)
        self.gradients, _grad_norm = tf.clip_by_global_norm(self.gradients, self.max_gradient_norm)
        self.gradients = list(zip(self.gradients, self.model_params))

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
        self.train = self.trainer.apply_gradients(self.gradients)

        self.train = tf.train.AdamOptimizer(self.learning_rate, name='optimizer').minimize(
            self.total_loss)

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype
