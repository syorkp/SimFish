import tensorflow.compat.v1 as tf
from Networks.utils import linear

from Networks.base_network import BaseNetwork
from Networks.Distributions.reflected_continuous import ReflectedProbabilityDist

tf.disable_v2_behavior()

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


class CategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from categorical input

        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits
        super(CategoricalProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=tf.stop_gradient(one_hot_actions))

    def kl(self, other):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a_1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        exp_a_1 = tf.exp(a_1)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        z_1 = tf.reduce_sum(exp_a_1, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (a_0 - tf.log(z_0) - a_1 + tf.log(z_1)), axis=-1)

    def entropy(self):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), axis=-1)

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        uniform = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(uniform)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the categorical logits input
        :return: (ProbabilityDistribution) the instance from the given categorical input
        """
        return cls(flat)


class PPONetworkActorDiscreteEmulator(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, clip_param, num_actions, epsilon_greedy=False):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=1)

        #            ----------        Non-Reflected       ---------            #

        self.action_stream, self.value_stream = tf.split(self.rnn_output, 2, 1)

        #            ----------        Reflected       ---------            #

        self.action_stream_ref, self.value_stream_ref = tf.split(self.rnn_output_ref, 2, 1)

        #            ----------        Combined       ---------            #

        pdparam_1 = linear(self.action_stream, 'pi', num_actions, init_scale=0.01, init_bias=0.0)
        pdparam_2 = linear(self.action_stream_ref, 'pi', num_actions, init_scale=0.01, init_bias=0.0)

        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = tf.split(pdparam_2, 10, 1)
        pdparam_2 = tf.concat([a0, a2, a1, a3, a5, a4, a6, a8, a7, a9], axis=1)
        pdparam = tf.divide(tf.add(pdparam_1, pdparam_2), 2)
        self.action_distribution = CategoricalProbabilityDistribution(pdparam)

        self.action_output = self.action_distribution.sample()

        self.neg_log_prob = self.action_distribution.neglogp(self.action_output)

        #            ----------        Value Outputs       ----------           #

        self.value_fn_1 = tf.layers.dense(self.value_stream, 1, name='vf')
        self.value_fn_2 = tf.layers.dense(self.value_stream_ref, 1, name='vf', reuse=True)

        self.value_output = tf.math.divide(tf.math.add(self.value_fn_1, self.value_fn_2), 2)

        #            ----------        Loss functions       ---------            #

        # Actor loss
        self.action_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='action_placeholder')

        self.new_neg_log_prob = self.action_distribution.neglogp(self.action_placeholder)
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
        self.entropy = tf.reduce_mean(self.action_distribution.entropy())  # TODO: works with new distribution?

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


    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower