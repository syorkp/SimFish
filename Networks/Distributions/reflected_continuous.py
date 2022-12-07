import tensorflow.compat.v1 as tf

from Networks.utils import linear

from Networks.Distributions.base_distributions import DiagGaussianProbabilityDistributionType

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class ReflectedProbabilityDist(DiagGaussianProbabilityDistributionType):

    def __init__(self, size):
        super().__init__(size)

    def proba_distribution_from_latent(self, policy_rnn_output, policy_rnn_output_ref, value_rnn_output,
                                       value_rnn_output_ref, init_scale=1.0, init_bias=0.0):
        mean_1 = linear(policy_rnn_output, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        mean_2 = linear(policy_rnn_output_ref, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)

        # Splitting via action
        mean_1_imp, mean_1_ang = tf.split(mean_1, 2, 1)
        mean_2_imp, mean_2_ang = tf.split(mean_2, 2, 1)

        # Combining
        mean_imp = tf.divide(tf.add(mean_1_imp, mean_2_imp), 2)
        mean_ang = tf.divide(tf.subtract(mean_1_ang, mean_2_ang), 2)

        mean = tf.concat([mean_imp, mean_ang], axis=1)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())

        # Combine
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

        q_values_1 = linear(value_rnn_output, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        q_values_2 = linear(value_rnn_output_ref, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        q_values = (q_values_1 + q_values_2)/2

        return self.proba_distribution_from_flat(pdparam), mean, logstd, q_values
