import tensorflow.compat.v1 as tf
import numpy as np


class BetaNormalDistribution:

    def __init__(self, concentration1, concentration0, mu, std):
        self.concentration1 = concentration1
        self.concentration0 = concentration0
        self.total_concentration = concentration0 + concentration1

        self.mu = mu
        self.std = std

        self.force_probs_to_zero_outside_support = True

    def sample(self, shape):
        return tf.concat([self.beta_sample(), self.normal_sample()], axis=1)

    def prob(self, x_beta, x_normal):
        return self.normal_prob(x_normal) * self.beta_prob(x_beta)

    def beta_sample(self):
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
        samples = tf.clip_by_value(samples, clip_value_min=0.01, clip_value_max=0.99)
        samples = tf.expand_dims(samples, 1)
        return samples

    def beta_log_prob(self, x):
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

    def beta_prob(self, x):
        return tf.exp(self.beta_log_prob(x))

    def beta_entropy(self):
        c1 = tf.reshape(self.concentration1, [-1])
        c0 = tf.reshape(self.concentration0, [-1])

        numerator = tf.math.exp(tf.math.lgamma(tf.math.add(c1, c0)))
        denominator = tf.math.multiply(tf.math.exp(tf.math.lgamma(c1)), tf.math.exp(tf.math.lgamma(c0)))
        normalisation_constant = tf.divide(numerator, denominator)

        entropy = tf.math.log(normalisation_constant) - ((c1 - 1)*tf.math.digamma(c1)) - ((c0 - 1)*tf.math.digamma(c0)) \
                  + ((c1 + c0 - 2) * tf.math.digamma(c1 + c0))

        return entropy

    def normal_neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mu) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(tf.math.log(self.std), axis=-1)

    def normal_prob(self, x):
        return tf.exp(-self.normal_neglogp(x))

    def kl(self, other):
        return tf.reduce_sum(other.logstd - tf.math.log(self.std) + (tf.square(self.std) + tf.square(self.mu - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def normal_entropy(self):
        return tf.reduce_sum(tf.math.log(self.std) + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def normal_sample(self):
        # Bounds are taken into account outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        return self.mu + self.std * tf.random_normal(tf.shape(self.mu),
                                                       dtype=self.mu.dtype)

    def entropy(self):
        return self.beta_entropy() * self.normal_entropy()

