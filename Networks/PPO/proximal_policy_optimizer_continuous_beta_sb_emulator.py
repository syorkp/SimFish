import tensorflow.compat.v1 as tf

from Networks.base_network import BaseNetwork
from Networks.Distributions.masked_multivariate_normal import MaskedMultivariateNormal
from Networks.Distributions.beta_normal_mix import BetaNormalDistribution
from Networks.Distributions.masked_beta_normal_mix import MaskedBetaNormalDistribution

tf.disable_v2_behavior()


class PPONetworkActorMultivariateBetaNormal2(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, max_impulse, max_angle_change,
                 clip_param, input_sigmas=False, impulse_scaling=None,
                 angle_scaling=None, max_sigma=1):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=2)

        #            ----------        Stream Splitting       ---aa------            #

        self.action_stream, self.value_stream = tf.split(self.rnn_output, 2, 1)
        self.action_stream_ref, self.value_stream_ref = tf.split(self.rnn_output_ref, 2, 1)

        if input_sigmas:
            self.impulse_stream, self.angle_stream = tf.split(self.action_stream, 2, 1)
            self.impulse_stream_ref, self.angle_stream_ref = tf.split(self.action_stream_ref, 2, 1)
        else:
            self.impulse_stream, self.impulse_stream_sigma, self.angle_stream, self.angle_stream_sigma = tf.split(
                self.action_stream, 4, 1)
            self.impulse_stream_ref, self.impulse_stream_sigma_ref, self.angle_stream_ref, self.angle_stream_sigma_ref = tf.split(
                self.action_stream_ref, 4, 1)

        #            ----------        Mu Estimations       ---------            #

        self.impulse_stream_1, self.impulse_stream_2 = tf.split(self.impulse_stream, 2, 1)
        self.impulse_stream_1_ref, self.impulse_stream_2_ref = tf.split(self.impulse_stream_ref, 2, 1)

        self.mu_impulse_1 = tf.layers.dense(self.impulse_stream_1, 1, activation=tf.nn.sigmoid,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          name=my_scope + '_mu_impulse_1', trainable=True)
        self.mu_impulse_2 = tf.layers.dense(self.impulse_stream_2, 1, activation=tf.nn.sigmoid,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          name=my_scope + '_mu_impulse_2', trainable=True)
        self.mu_impulse_1_ref = tf.layers.dense(self.impulse_stream_1_ref, 1, activation=tf.nn.sigmoid,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              name=my_scope + '_mu_impulse_1', trainable=True, reuse=True)
        self.mu_impulse_2_ref = tf.layers.dense(self.impulse_stream_2_ref, 1, activation=tf.nn.sigmoid,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              name=my_scope + '_mu_impulse_2', trainable=True, reuse=True)

        self.mu_angle = tf.layers.dense(self.angle_stream, 1, activation=tf.nn.tanh,
                                        kernel_initializer=tf.orthogonal_initializer,
                                        name=my_scope + '_mu_angle', trainable=True)
        self.mu_angle_ref = tf.layers.dense(self.angle_stream_ref, 1, activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer,
                                            name=my_scope + '_mu_angle', trainable=True, reuse=True)

        # Combining
        self.mu_impulse_1_combined = tf.math.divide(tf.math.add(self.mu_impulse_1, self.mu_impulse_1_ref), 2.0,
                                                    name="mu_impulse_1_combined")
        self.mu_impulse_1_combined = self.bounded_output(self.mu_impulse_1_combined, 0, 1)

        self.mu_impulse_2_combined = tf.math.divide(tf.math.add(self.mu_impulse_2, self.mu_impulse_2_ref), 2.0,
                                                    name="mu_impulse_2_combined")
        self.mu_impulse_2_combined = self.bounded_output(self.mu_impulse_2_combined, 0, 1)

        # For logging purposes:
        self.mu_impulse = self.mu_impulse_1_combined
        self.mu_impulse_ref = self.mu_impulse_2_combined
        self.mu_impulse_combined = tf.divide(self.mu_impulse_1_combined, tf.add(self.mu_impulse_1_combined, self.mu_impulse_2_combined))

        self.mu_angle_combined = tf.divide(tf.subtract(self.mu_angle, self.mu_angle_ref), 2)

        self.mu_action = tf.concat([self.mu_impulse_combined, self.mu_angle_combined], axis=1)

        #            ----------        Sigma Estimations       ---------            #

        self.sigma_impulse_combined_proto = tf.placeholder(shape=[None], dtype=tf.float32,
                                                           name='sigma_impulse_combined')
        self.sigma_angle_combined_proto = tf.placeholder(shape=[None], dtype=tf.float32, name='sigma_angle_combined')

        if input_sigmas:
            self.sigma_impulse_combined = tf.expand_dims(self.sigma_impulse_combined_proto, 1)
            self.sigma_angle_combined = tf.expand_dims(self.sigma_angle_combined_proto, 1)

        else:
            self.sigma_impulse = tf.layers.dense(self.impulse_stream_sigma, 1, activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 name=my_scope + '_sigma_impulse', trainable=True)
            self.sigma_impulse = self.bounded_output(self.sigma_impulse, 0, max_sigma)

            self.sigma_angle = tf.layers.dense(self.angle_stream_sigma, 1, activation=tf.nn.sigmoid,
                                               kernel_initializer=tf.orthogonal_initializer,
                                               name=my_scope + '_sigma_angle', trainable=True)
            self.sigma_angle = self.bounded_output(self.sigma_angle, 0, max_sigma)

            self.sigma_impulse_ref = tf.layers.dense(self.impulse_stream_sigma_ref, 1, activation=tf.nn.sigmoid,
                                                     kernel_initializer=tf.orthogonal_initializer,
                                                     name=my_scope + '_sigma_impulse', trainable=True, reuse=True)
            self.sigma_impulse_ref = self.bounded_output(self.sigma_impulse_ref, 0, max_sigma)

            self.sigma_angle_ref = tf.layers.dense(self.angle_stream_sigma_ref, 1, activation=tf.nn.sigmoid,
                                                   kernel_initializer=tf.orthogonal_initializer,
                                                   name=my_scope + '_sigma_angle', trainable=True, reuse=True)
            self.sigma_angle_ref = self.bounded_output(self.sigma_angle_ref, 0, max_sigma)

            self.sigma_impulse_combined = tf.divide(tf.add(self.sigma_impulse, self.sigma_impulse_ref), 2)
            self.sigma_angle_combined = tf.divide(tf.add(self.sigma_angle, self.sigma_angle_ref), 2)

        self.sigma_action = tf.concat([self.sigma_impulse_combined, self.sigma_angle_combined], axis=1)

        # self.log_std = tf.get_variable(name='logstd', shape=[1, 2], initializer=tf.zeros_initializer(), trainable=True)
        # self.sigma_action = tf.exp(self.log_std)

        #            ----------        Form Distribution Estimations       ---------            #

        if impose_action_mask:
            self.action_distribution = MaskedBetaNormalDistribution(self.mu_impulse_1_combined, self.mu_impulse_2_combined,
                                                                    self.mu_angle_combined, self.sigma_angle_combined,
                                                                    impulse_scaling=max_impulse,
                                                                    angle_scaling=max_angle_change
                                                                    )
            self.action_output = self.action_distribution.sample_masked(1)
            # self.action_output = tf.squeeze(self.action_distribution.sample(1), axis=0)
            # TODO: Build in action mask
            # self.action_distribution = BetaNormalDistribution(loc=self.mu_action, scale_diag=self.sigma_action,
            #                                                     impulse_scaling=max_impulse,
            #                                                     angle_scaling=max_angle_change)
            # self.action_output = tf.squeeze(self.action_distribution.sample_masked(1), axis=0)

        else:
            self.action_distribution = BetaNormalDistribution(self.mu_impulse_1_combined, self.mu_impulse_2_combined,
                                                                  self.mu_angle_combined, self.sigma_angle_combined)

            self.action_output = self.action_distribution.sample(1)

        self.impulse_output_unscaled, self.angle_output_unscaled = tf.split(self.action_output, 2, axis=1)

        if impose_action_mask:
            self.impulse_output = tf.math.multiply(self.impulse_output_unscaled, max_impulse, name="impulse_output")
            self.angle_output = tf.math.multiply(self.angle_output_unscaled, max_angle_change, name="angle_output")
        else:
            self.impulse_output = tf.clip_by_value(self.impulse_output_unscaled, 0, 1)
            self.angle_output = tf.clip_by_value(self.angle_output_unscaled, -1, 1)

            self.impulse_output = tf.math.multiply(self.impulse_output, max_impulse, name="impulse_output")
            self.angle_output = tf.math.multiply(self.angle_output, max_angle_change, name="angle_output")

        self.neg_log_prob = -tf.math.log(
            self.action_distribution.prob(self.impulse_output_unscaled, self.angle_output_unscaled))

        #            ----------        Value Outputs       ----------           #

        self.value_fn_1 = tf.layers.dense(self.value_stream, 1, name='vf')
        self.value_fn_2 = tf.layers.dense(self.value_stream_ref, 1, name='vf', reuse=True)

        self.value_output = tf.math.divide(tf.math.add(self.value_fn_1, self.value_fn_2), 2)

        #            ----------        Loss functions       ---------            #

        # Actor loss
        self.action_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='action_placeholder')
        self.impulse_placeholder, self.angle_placeholder = tf.split(self.action_placeholder, 2, axis=1)

        if impose_action_mask:
            self.impulse_placeholder = tf.math.divide(self.impulse_placeholder, max_impulse)
            self.angle_placeholder = tf.math.divide(self.angle_placeholder, max_angle_change)
        else:
            self.impulse_placeholder = tf.math.divide(self.impulse_placeholder, max_impulse)
            self.angle_placeholder = tf.math.divide(self.angle_placeholder, max_angle_change)

        self.normalised_action = tf.concat([self.impulse_placeholder, self.angle_placeholder], axis=1)

        self.new_log_prob = self.action_distribution.prob(tf.divide(self.impulse_placeholder, max_impulse),
                                                              tf.divide(self.angle_placeholder, max_angle_change))
        self.maxnli = tf.math.reduce_max(self.new_log_prob) + 1
        self.new_log_prob = tf.math.divide(self.new_log_prob, self.maxnli)
        self.new_neg_log_prob = -tf.math.log(self.new_log_prob)

        self.old_neg_log_prob = tf.placeholder(shape=[None], dtype=tf.float32, name='old_log_prob_impulse')
        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')

        self.ratio = tf.exp(self.old_neg_log_prob - self.new_neg_log_prob)
        self.surrogate_loss_1 = -tf.math.multiply(self.ratio, self.scaled_advantage_placeholder)
        self.surrogate_loss_2 = -tf.math.multiply(
            tf.clip_by_value(self.ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)
        self.policy_loss = tf.reduce_mean(tf.maximum(self.surrogate_loss_1, self.surrogate_loss_2))

        # Value loss
        self.returns_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='returns')
        self.old_value_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value')
        # self.value_cliprange_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='value_cliprange')
        # Clip the different between old and new value NOTE: this depends on the reward scaling
        self.value_clipped = self.old_value_placeholder + tf.clip_by_value(
            self.value_output - self.old_value_placeholder, -clip_param, clip_param)

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
        # self.total_loss = self.policy_loss + tf.multiply(self.value_loss, self.value_coefficient)
        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        # Gradient clipping (for stability)
        self.model_params = tf.trainable_variables()
        self.model_gradients = tf.gradients(self.total_loss, self.model_params)
        self.model_gradients, _grad_norm = tf.clip_by_global_norm(self.model_gradients, self.max_gradient_norm)
        self.model_gradients = list(zip(self.model_gradients, self.model_params))

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
        self.train = self.trainer.apply_gradients(self.model_gradients)

        # TODO: Probably not meant to be there, but changed since main tests.
        # self.train = tf.train.AdamOptimizer(self.learning_rate, name='optimizer').minimize(
        #     self.total_loss)  # Two trains?

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower

