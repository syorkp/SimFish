# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class ProbabilityDistribution(object):
    """
    Base class for describing a probability distribution.
    """
    def __init__(self):
        super(ProbabilityDistribution, self).__init__()

    def flatparam(self):
        """
        Return the direct probabilities

        :return: ([float]) the probabilities
        """
        raise NotImplementedError

    def mode(self):
        """
        Returns the probability

        :return: (Tensorflow Tensor) the deterministic action
        """
        raise NotImplementedError

    def neglogp(self, x):
        """
        returns the of the negative log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The negative log likelihood of the distribution
        """
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        """
        Calculates the Kullback-Leibler divergence from the given probability distribution

        :param other: ([float]) the distribution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns Shannon's entropy of the probability

        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probability distribution

        :return: (Tensorflow Tensor) the stochastic action
        """
        raise NotImplementedError

    def logp(self, x):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x)


class ProbabilityDistributionType(object):
    """
    Parametrized family of probability distributions
    """

    def probability_distribution_class(self):
        """
        returns the ProbabilityDistribution class of this type

        :return: (Type ProbabilityDistribution) the probability distribution class associated
        """
        raise NotImplementedError

    def proba_distribution_from_flat(self, flat):
        """
        Returns the probability distribution from flat probabilities
        flat: flattened vector of parameters of probability distribution

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def param_shape(self):
        """
        returns the shape of the input parameters

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_shape(self):
        """
        returns the shape of the sampling

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_dtype(self):
        """
        returns the type of the sampling

        :return: (type) the type
        """
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the input parameters

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the sampling

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)


class DiagGaussianProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        super(DiagGaussianProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        # Bounds are taken into account outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean),
                                                       dtype=self.mean.dtype)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate Gaussian input data
        """
        return cls(flat)


class DiagGaussianProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for multivariate Gaussian input

        :param size: (int) the number of dimensions of the multivariate gaussian
        """
        self.size = size

    def probability_distribution_class(self):
        return DiagGaussianProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32

