import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from Networks.base_network import BaseNetwork
from Networks.Distributions.masked_multivariate_normal import MaskedMultivariateNormal

tf.disable_v2_behavior()


class PPONetworkActorMultivariate(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, max_impulse, max_angle_change,
                 clip_param, input_sigmas=False, new_simulation=True, impose_action_mask=False, impulse_scaling=None, angle_scaling=None):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=2, new_simulation=new_simulation)

        #            ----------        Stream Splitting       ---------            #

        if input_sigmas:
            self.mu_impulse_stream, self.mu_angle_stream = tf.split(self.rnn_output, 2, 1)
            self.mu_impulse_stream_ref, self.mu_angle_stream_ref = tf.split(self.rnn_output_ref, 2, 1)
        else:
            self.mu_impulse_stream, self.sigma_impulse_stream, self.mu_angle_stream, self.sigma_angle_stream = tf.split(self.rnn_output, 4, 1)
            self.mu_impulse_stream_ref, self.sigma_impulse_stream_ref, self.mu_angle_stream_ref, self.sigma_angle_stream_ref = tf.split(self.rnn_output_ref, 4, 1)

        #            ----------        Mu Estimations       ---------            #


        self.mu_impulse = tf.layers.dense(self.mu_impulse_stream, 1, activation=tf.nn.sigmoid,
                                          kernel_initializer=tf.orthogonal_initializer,
                                          name=my_scope + '_mu_impulse', trainable=True)

        self.mu_angle = tf.layers.dense(self.mu_angle_stream, 1, activation=tf.nn.tanh,
                                        kernel_initializer=tf.orthogonal_initializer,
                                        name=my_scope + '_mu_angle', trainable=True)

        self.mu_impulse_ref = tf.layers.dense(self.mu_impulse_stream_ref, 1, activation=tf.nn.sigmoid,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              name=my_scope + '_mu_impulse', reuse=True, trainable=True)

        self.mu_angle_ref = tf.layers.dense(self.mu_angle_stream_ref, 1, activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer,
                                            name=my_scope + '_mu_angle', reuse=True, trainable=True)

        self.mu_impulse_combined = tf.math.divide(tf.math.add(self.mu_impulse, self.mu_impulse_ref), 2.0,
                                                  name="mu_impulse_combined")
        self.mu_angle_combined = tf.math.divide(tf.math.subtract(self.mu_angle, self.mu_angle_ref), 2.0,
                                                name="mu_angle_combined")

        self.mu_action = tf.concat([self.mu_impulse_combined, self.mu_angle_combined], axis=1)

        #            ----------        Sigma Estimations       ---------            #

        self.sigma_impulse_combined_proto = tf.placeholder(shape=[None], dtype=tf.float32,
                                                           name='sigma_impulse_combined')
        self.sigma_angle_combined_proto = tf.placeholder(shape=[None], dtype=tf.float32, name='sigma_angle_combined')

        if input_sigmas:
            self.sigma_impulse_combined = tf.expand_dims(self.sigma_impulse_combined_proto, 1)
            self.sigma_angle_combined = tf.expand_dims(self.sigma_angle_combined_proto, 1)

        else:
            self.sigma_impulse = tf.layers.dense(self.sigma_impulse_stream, 1,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 name=my_scope + '_sigma_impulse', trainable=True)
            self.sigma_impulse = self.bounded_output(self.sigma_impulse, 0, 1)

            self.sigma_impulse_ref = tf.layers.dense(self.sigma_impulse_stream_ref, 1,
                                                     kernel_initializer=tf.orthogonal_initializer,
                                                     name=my_scope + '_sigma_impulse', reuse=True, trainable=True)
            self.sigma_impulse_ref = self.bounded_output(self.sigma_impulse_ref, 0, 1)

            self.sigma_angle = tf.layers.dense(self.sigma_angle_stream, 1,
                                               kernel_initializer=tf.orthogonal_initializer,
                                               name=my_scope + '_sigma_angle', trainable=True)
            self.sigma_angle = self.bounded_output(self.sigma_angle, 0, 1)

            self.sigma_angle_ref = tf.layers.dense(self.sigma_angle_stream_ref, 1,
                                                   kernel_initializer=tf.orthogonal_initializer,
                                                   name=my_scope + '_sigma_angle', reuse=True, trainable=True)
            self.sigma_angle_ref = self.bounded_output(self.sigma_angle_ref, 0, 1)

            self.sigma_impulse_combined = tf.math.divide(tf.math.add(self.sigma_impulse, self.sigma_impulse_ref), 2.0,
                                                         name="sigma_impulse_combined")
            self.sigma_angle_combined = tf.math.divide(tf.math.add(self.sigma_angle, self.sigma_angle_ref), 2.0,
                                                       name="sigma_angle_combined")

        self.sigma_action = tf.concat([self.sigma_impulse_combined, self.sigma_angle_combined], axis=1)

        #            ----------        Form Distribution Estimations       ---------            #

        # Multinomial distribution
        if impose_action_mask:
            self.action_distribution = MaskedMultivariateNormal(loc=self.mu_action, scale_diag=self.sigma_action,
                                                                impulse_scaling=impulse_scaling,
                                                                angle_scaling=angle_scaling)
            self.action_output = tf.squeeze(self.action_distribution.sample_masked(1), axis=0)

        else:
            self.action_distribution = tfp.distributions.MultivariateNormalDiag(loc=self.mu_action,
                                                                                scale_diag=self.sigma_action)
            self.action_output = tf.squeeze(self.action_distribution.sample(1), axis=0)

        self.impulse_output, self.angle_output = tf.split(self.action_output, 2, axis=1)

        if impose_action_mask:
            self.impulse_output = tf.math.multiply(self.impulse_output, impulse_scaling, name="impulse_output")
            self.angle_output = tf.math.multiply(self.angle_output, angle_scaling, name="angle_output")
        else:
            self.impulse_output = tf.clip_by_value(self.impulse_output, 0, 1)
            self.angle_output = tf.clip_by_value(self.angle_output, -1, 1)

            self.impulse_output = tf.math.multiply(self.impulse_output, max_impulse, name="impulse_output")
            self.angle_output = tf.math.multiply(self.angle_output, max_angle_change, name="angle_output")

        self.log_prob = tf.log(self.action_distribution.prob(self.action_output) + 1e-5)

        #            ----------        Loss functions       ---------            #

        # Placeholders
        self.action_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='impulse_placeholder')
        self.impulse_placeholder, self.angle_placeholder = tf.split(self.action_placeholder, 2, axis=1)

        if impose_action_mask:
            self.impulse_placeholder = tf.math.divide(self.impulse_placeholder, impulse_scaling)
            self.angle_placeholder = tf.math.divide(self.angle_placeholder, angle_scaling)
        else:
            self.impulse_placeholder = tf.math.divide(self.impulse_placeholder, max_impulse)
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

