import tensorflow.compat.v1 as tf
from tensorflow_probability.python import math as tfp_math


class MySimpleBetaDistribution:

    def __init__(self, concentration1, concentration0):
        self.concentration1 = concentration1
        self.concentration0 = concentration0
        self.total_concentration = concentration0 + concentration1
        self.force_probs_to_zero_outside_support = True

    def sample(self, shape):
        """
        Samples according to Johnk's method
        :param shape:
        :return:
        """
        c1 = tf.reshape(self.concentration1, [-1])
        c0 = tf.reshape(self.concentration0, [-1])
        U1 = tf.random.uniform((), 0, 1)
        U2 = tf.random.uniform((), 0, 1)
        V1 = tf.math.pow(U1, tf.math.divide(1, c1))
        V2 = tf.math.pow(U2, tf.math.divide(1, c0))

        W = tf.math.add(V1, V2)
        s1 = tf.math.divide(V1, tf.math.add(V1, V2))

        logx = tf.math.divide(tf.math.log(U1), c1)
        logy = tf.math.divide(tf.math.log(U2), c0)
        logm = tf.where(logx > logy, x=logx, y=logy)
        logx -= logm
        logy -= logm
        s2 = tf.math.exp(logx - tf.math.log(tf.math.add(tf.math.exp(logx), tf.math.exp(logy))))

        samples = tf.where(W <= 1, x=s1, y=s2)
        samples = tf.expand_dims(samples, 1)
        return samples
        # seed1, seed2 = samplers.split_seed(None, salt='beta')
        # concentration1 = tf.convert_to_tensor(self.concentration1)
        # concentration0 = tf.convert_to_tensor(self.concentration0)
        # log_gamma1 = gamma_lib.random_gamma(
        #     shape=[n], concentration=concentration1, seed=seed1,
        #     log_space=True)
        # log_gamma2 = gamma_lib.random_gamma(
        #     shape=[n], concentration=concentration0, seed=seed2,
        #     log_space=True)
        # return tf.math.sigmoid(log_gamma1 - log_gamma2)

    # def log_prob(self, x):
    #     concentration0 = tf.convert_to_tensor(self.concentration0)
    #     concentration1 = tf.convert_to_tensor(self.concentration1)
    #     lp = (self._log_unnormalized_prob(x, concentration1, concentration0) -
    #           self._log_normalization(concentration1, concentration0))
    #     # if self.force_probs_to_zero_outside_support:
    #     #     return tf.where((x >= 0) & (x <= 1), lp, -float('inf'))
    #     return lp

    def log_prob(self, x):
        c1 = tf.reshape(self.concentration1, [-1])
        c0 = tf.reshape(self.concentration0, [-1])
        x = tf.reshape(x, [-1])

        unnormalised_prob1 = tf.math.pow(x, tf.math.subtract(c1, 1.0))
        unnormalised_prob2 = tf.math.pow(tf.math.subtract(1.0, x), tf.math.subtract(c0, 1.0))
        unnormalised_prob = tf.math.multiply(unnormalised_prob1, unnormalised_prob2)

        numerator = tf.math.exp(tf.math.lgamma(tf.math.add(c1, c0)))
        denominator = tf.math.multiply(tf.math.exp(tf.math.lgamma(c1)), tf.math.exp(tf.math.lgamma(c0)))
        normalisation_constant = tf.divide(numerator, denominator)

        normalised_prob = tf.math.multiply(unnormalised_prob, normalisation_constant)
        log_normalised_prob = tf.math.log(normalised_prob)
        log_normalised_prob = tf.expand_dims(log_normalised_prob, 1)
        return log_normalised_prob

    def prob(self, x):
        return tf.exp(self.log_prob(x))

    def entropy(self):
        c1 = tf.reshape(self.concentration1, [-1])
        c0 = tf.reshape(self.concentration0, [-1])

        numerator = tf.math.exp(tf.math.lgamma(tf.math.add(c1, c0)))
        denominator = tf.math.multiply(tf.math.exp(tf.math.lgamma(c1)), tf.math.exp(tf.math.lgamma(c0)))
        normalisation_constant = tf.divide(numerator, denominator)

        entropy = tf.math.log(normalisation_constant) - ((c1 - 1)*tf.math.digamma(c1)) - ((c0 - 1)*tf.math.digamma(c0)) \
                  + ((c1 + c0 - 2) * tf.math.digamma(c1 + c0))

        return entropy

        # concentration1 = tf.convert_to_tensor(self.concentration1)
        # concentration0 = tf.convert_to_tensor(self.concentration0)
        # total_concentration = concentration1 + concentration0
        # return (self._log_normalization(concentration1, concentration0) -
        #         (concentration1 - 1.) * tf.math.digamma(concentration1) -
        #         (concentration0 - 1.) * tf.math.digamma(concentration0) +
        #         (total_concentration - 2.) * tf.math.digamma(total_concentration))


    # def _log_normalization(self, concentration1, concentration0):
    #     return tfp_math.lbeta(concentration1, concentration0)
    #
    # def _log_unnormalized_prob(self, x, concentration1, concentration0):
    #     return (tf.math.xlogy(concentration1 - 1., x) +
    #             tf.math.xlog1py(concentration0 - 1., -x))


