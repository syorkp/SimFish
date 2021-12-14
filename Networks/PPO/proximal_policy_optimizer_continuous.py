import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from Networks.base_network import BaseNetwork
from Networks.Distributions.my_simple_beta_distribution import BetaDistribution
from Networks.Distributions.beta_normal_mix import BetaNormalDistribution

tf.disable_v2_behavior()


class PPONetworkActor(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, max_impulse, max_angle_change,
                 clip_param, beta_impulse=False, new_simulation=True):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=2, new_simulation=new_simulation)

        self.mu_impulse_stream, self.mu_angle_stream = tf.split(self.rnn_output, 2, 1)
        self.mu_impulse_stream_ref, self.mu_angle_stream_ref = tf.split(self.rnn_output_ref, 2, 1)

        self.sigma_impulse_combined = tf.placeholder(shape=[None], dtype=tf.float32, name='sigma_impulse_combined')

        if beta_impulse:
            self.mu_impulse_stream_1, self.mu_impulse_stream_2 = tf.split(self.mu_impulse_stream, 2, 1)
            self.mu_impulse_stream_1_ref, self.mu_impulse_stream_2_ref = tf.split(self.mu_impulse_stream_ref, 2, 1)

            self.mu_impulse_1 = tf.layers.dense(self.mu_impulse_stream_1, 1, activation=tf.nn.sigmoid,
                                                kernel_initializer=tf.orthogonal_initializer,
                                                name=my_scope + '_mu_impulse_1', trainable=True)
            self.mu_impulse_2 = tf.layers.dense(self.mu_impulse_stream_2, 1, activation=tf.nn.sigmoid,
                                                kernel_initializer=tf.orthogonal_initializer,
                                                name=my_scope + '_mu_impulse_2', trainable=True)

            self.mu_impulse_1_ref = tf.layers.dense(self.mu_impulse_stream_1_ref, 1, activation=tf.nn.sigmoid,
                                                    kernel_initializer=tf.orthogonal_initializer,
                                                    name=my_scope + '_mu_impulse_1', trainable=True, reuse=True)
            self.mu_impulse_2_ref = tf.layers.dense(self.mu_impulse_stream_2_ref, 1, activation=tf.nn.sigmoid,
                                                    kernel_initializer=tf.orthogonal_initializer,
                                                    name=my_scope + '_mu_impulse_2', trainable=True, reuse=True)

            self.mu_impulse_1_combined = tf.math.divide(tf.math.add(self.mu_impulse_1, self.mu_impulse_1_ref), 2.0,
                                                        name="mu_impulse_1_combined")
            self.mu_impulse_1_combined = self.bounded_output(self.mu_impulse_1_combined, 0, 1)

            self.mu_impulse_2_combined = tf.math.divide(tf.math.add(self.mu_impulse_2, self.mu_impulse_2_ref), 2.0,
                                                        name="mu_impulse_2_combined")
            self.mu_impulse_2_combined = self.bounded_output(self.mu_impulse_2_combined, 0, 1)

            # For logging purposes:
            self.mu_impulse = self.mu_impulse_1_combined
            self.mu_impulse_ref = self.mu_impulse_2_combined
            self.mu_impulse_combined = tf.divide(tf.add(self.mu_impulse_1_combined, self.mu_impulse_2_combined), 2)

            # self.norm_dist_impulse = BetaDistribution(self.mu_impulse_1_combined, self.mu_impulse_2_combined)

            # Additions:
            self.mu_angle = tf.layers.dense(self.mu_angle_stream, 1, activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_mu_angle',
                                            trainable=True)
            self.mu_angle_ref = tf.layers.dense(self.mu_angle_stream_ref, 1, activation=tf.nn.tanh,
                                                kernel_initializer=tf.orthogonal_initializer,
                                                name=my_scope + '_mu_angle', reuse=True, trainable=True)
            self.mu_angle_combined = tf.math.divide(tf.math.subtract(self.mu_angle, self.mu_angle_ref), 2.0,
                                                    name="mu_angle_combined")
            self.sigma_angle_combined = tf.placeholder(shape=[None], dtype=tf.float32, name='sigma_angle_combined')

            self.norm_dist_impulse_angle = BetaNormalDistribution(self.mu_impulse_1_combined, self.mu_impulse_2_combined,
                                                                  self.mu_angle_combined, self.sigma_angle_combined)

            self.action_tf_var_impulse_angle = self.norm_dist_impulse_angle.sample(1)
            self.action_tf_var_impulse, self.action_tf_var_angle = tf.split(self.action_tf_var_impulse_angle, 2, 1)

            self.impulse_output = tf.math.multiply(self.action_tf_var_impulse, max_impulse, name="impulse_output")

            self.action_tf_var_angle = tf.clip_by_value(self.action_tf_var_angle, -1, 1)
            self.angle_output = tf.math.multiply(self.action_tf_var_angle, max_angle_change, name="angle_output")

            self.log_prob_impulse = tf.math.log(self.norm_dist_impulse_angle.prob(self.action_tf_var_impulse, self.action_tf_var_angle))
            self.log_prob_angle = self.log_prob_impulse
        else:
            self.mu_impulse = tf.layers.dense(self.mu_impulse_stream, 1, activation=tf.nn.sigmoid,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              name=my_scope + '_mu_impulse', trainable=True)

            self.mu_impulse_ref = tf.layers.dense(self.mu_impulse_stream_ref, 1, activation=tf.nn.sigmoid,
                                                  kernel_initializer=tf.orthogonal_initializer,
                                                  name=my_scope + '_mu_impulse', reuse=True, trainable=True)

            self.mu_impulse_combined = tf.math.divide(tf.math.add(self.mu_impulse, self.mu_impulse_ref), 2.0,
                                                      name="mu_impulse_combined")

            self.norm_dist_impulse = tf.distributions.Normal(self.mu_impulse_combined, self.sigma_impulse_combined,
                                                             name="norm_dist_impulse")

            # Following should be outside:
            self.action_tf_var_impulse = tf.squeeze(self.norm_dist_impulse.sample(1), axis=0)
            # TODO: Shouldnt be clipping??
            self.impulse_output = tf.math.multiply(self.action_tf_var_impulse, max_impulse, name="impulse_output")
            self.log_prob_impulse = self.norm_dist_impulse.log_prob(self.action_tf_var_impulse)

            # Combined Actor angle output
            self.mu_angle = tf.layers.dense(self.mu_angle_stream, 1, activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_mu_angle',
                                            trainable=True)
            self.mu_angle_ref = tf.layers.dense(self.mu_angle_stream_ref, 1, activation=tf.nn.tanh,
                                                kernel_initializer=tf.orthogonal_initializer,
                                                name=my_scope + '_mu_angle', reuse=True, trainable=True)
            self.mu_angle_combined = tf.math.divide(tf.math.subtract(self.mu_angle, self.mu_angle_ref), 2.0,
                                                    name="mu_angle_combined")
            self.sigma_angle_combined = tf.placeholder(shape=[None], dtype=tf.float32, name='sigma_angle_combined')
            self.norm_dist_angle = tf.distributions.Normal(self.mu_angle_combined, self.sigma_angle_combined,
                                                           name="norm_dist_angle")
            self.action_tf_var_angle = tf.squeeze(self.norm_dist_angle.sample(1), axis=0)
            self.action_tf_var_angle = tf.clip_by_value(self.action_tf_var_angle, -1, 1)
            self.angle_output = tf.math.multiply(self.action_tf_var_angle, max_angle_change, name="angle_output")
            self.log_prob_angle = tf.log(self.norm_dist_angle.prob(self.action_tf_var_angle) + 1e-5)

        #            ----------        Loss functions       ---------            #

        # Placeholders
        self.impulse_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='impulse_placeholder')
        self.angle_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='angle_placeholder')

        if beta_impulse:
            self.new_log_prob = self.norm_dist_impulse_angle.prob(tf.divide(self.impulse_placeholder, max_impulse),
                                                                  tf.divide(self.angle_placeholder, max_angle_change))
            self.maxnli = tf.math.reduce_max(self.new_log_prob) + 1
            self.new_log_prob = tf.math.divide(self.new_log_prob, self.maxnli)
            self.new_log_prob = tf.math.log(self.new_log_prob)
        else:
            self.new_log_prob_impulse = self.norm_dist_impulse.log_prob(
                tf.math.divide(self.impulse_placeholder, max_impulse))
            self.new_log_prob_angle = tf.log(
                self.norm_dist_angle.prob(tf.math.divide(self.angle_placeholder, max_angle_change)) + 1e-5)

        self.old_log_prob_impulse_placeholder = tf.placeholder(shape=[None], dtype=tf.float32,
                                                               name='old_log_prob_impulse')
        self.old_log_prob_angle_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='old_log_prob_angle')

        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')

        # COMBINED LOSS

        if beta_impulse:
            self.ratio = tf.exp(self.new_log_prob - self.old_log_prob_impulse_placeholder)
            self.surrogate_loss_1 = tf.math.multiply(self.ratio, self.scaled_advantage_placeholder)
            self.surrogate_loss_2 = tf.math.multiply(
                tf.clip_by_value(self.ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)

            self.impulse_loss = -tf.reduce_mean(tf.minimum(self.surrogate_loss_1, self.surrogate_loss_2))
            self.angle_loss = -tf.reduce_mean(tf.minimum(self.surrogate_loss_1, self.surrogate_loss_2))
            self.total_loss = -tf.reduce_mean(tf.minimum(self.surrogate_loss_1, self.surrogate_loss_2))
        else:
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
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='actor_optimizer_impulse').minimize(
            self.total_loss)
        # self.max_gradient_norm = 0.5
        #
        # # Gradient clipping (for stability)
        # self.model_params = tf.trainable_variables()
        # self.model_gradients = tf.gradients(self.total_loss, self.model_params)
        # self.model_gradients, _grad_norm = tf.clip_by_global_norm(self.model_gradients, self.max_gradient_norm)
        # self.model_gradients = list(zip(self.model_gradients, self.model_params))
        #
        # self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
        # self.train = self.trainer.apply_gradients(self.model_gradients)

        # TODO: Probably not meant to be there.
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='optimizer').minimize(
        #     self.total_loss)

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower
