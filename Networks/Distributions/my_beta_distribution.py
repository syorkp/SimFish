import numpy as np
import six

from tensorflow.python.framework import tensor_util
from tensorflow.python.ops.distributions import util
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
import contextlib


class MyBetaDistribution:
    """Beta distribution.
    The Beta distribution is defined over the `(0, 1)` interval using parameters
    `concentration1` (aka "alpha") and `concentration0` (aka "beta").
    #### Mathematical Details
    The probability density function (pdf) is,
    ```none
    pdf(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
    Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
    ```
    where:
    * `concentration1 = alpha`,
    * `concentration0 = beta`,
    * `Z` is the normalization constant, and,
    * `Gamma` is the [gamma function](
      https://en.wikipedia.org/wiki/Gamma_function).
    The concentration parameters represent mean total counts of a `1` or a `0`,
    i.e.,
    ```none
    concentration1 = alpha = mean * total_concentration
    concentration0 = beta  = (1. - mean) * total_concentration
    ```
    where `mean` in `(0, 1)` and `total_concentration` is a positive real number
    representing a mean `total_count = concentration1 + concentration0`.
    Distribution parameters are automatically broadcast in all functions; see
    examples for details.
    Warning: The samples can be zero due to finite precision.
    This happens more often when some of the concentrations are very small.
    Make sure to round the samples to `np.finfo(dtype).tiny` before computing the
    density.
    Samples of this distribution are reparameterized (pathwise differentiable).
    The derivatives are computed using the approach described in
    (Figurnov et al., 2018).
    #### Examples
    ```python
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    # Create a batch of three Beta distributions.
    alpha = [1, 2, 3]
    beta = [1, 2, 3]
    dist = tfd.Beta(alpha, beta)
    dist.sample([4, 5])  # Shape [4, 5, 3]
    # `x` has three batch entries, each with two samples.
    x = [[.1, .4, .5],
         [.2, .3, .5]]
    # Calculate the probability of each pair of samples under the corresponding
    # distribution in `dist`.
    dist.prob(x)         # Shape [2, 3]
    ```
    ```python
    # Create batch_shape=[2, 3] via parameter broadcast:
    alpha = [[1.], [2]]      # Shape [2, 1]
    beta = [3., 4, 5]        # Shape [3]
    dist = tfd.Beta(alpha, beta)
    # alpha broadcast as: [[1., 1, 1,],
    #                      [2, 2, 2]]
    # beta broadcast as:  [[3., 4, 5],
    #                      [3, 4, 5]]
    # batch_Shape [2, 3]
    dist.sample([4, 5])  # Shape [4, 5, 2, 3]
    x = [.2, .3, .5]
    # x will be broadcast as [[.2, .3, .5],
    #                         [.2, .3, .5]],
    # thus matching batch_shape [2, 3].
    dist.prob(x)         # Shape [2, 3]
    ```
    Compute the gradients of samples w.r.t. the parameters:
    ```python
    alpha = tf.constant(1.0)
    beta = tf.constant(2.0)
    dist = tfd.Beta(alpha, beta)
    samples = dist.sample(5)  # Shape [5]
    loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
    # Unbiased stochastic gradients of the loss function
    grads = tf.gradients(loss, [alpha, beta])
    ```
    References:
      Implicit Reparameterization Gradients:
        [Figurnov et al., 2018]
        (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
        ([pdf]
        (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
    """

    def __init__(self,
                 concentration1=None,
                 concentration0=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="Beta"):
        """Initialize a batch of Beta distributions.
        Args:
          concentration1: Positive floating-point `Tensor` indicating mean
            number of successes; aka "alpha". Implies `self.dtype` and
            `self.batch_shape`, i.e.,
            `concentration1.shape = [N1, N2, ..., Nm] = self.batch_shape`.
          concentration0: Positive floating-point `Tensor` indicating mean
            number of failures; aka "beta". Otherwise has same semantics as
            `concentration1`.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
            (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
            result is undefined. When `False`, an exception is raised if one or
            more of the statistic's batch members are undefined.
          name: Python `str` name prefixed to Ops created by this class.
        """
        parameters = dict(locals())
        with ops.name_scope(name, values=[concentration1, concentration0]) as name:
            self._concentration1 = self._maybe_assert_valid_concentration(
                ops.convert_to_tensor(concentration1, name="concentration1"),
                validate_args)
            self._concentration0 = self._maybe_assert_valid_concentration(
                ops.convert_to_tensor(concentration0, name="concentration0"),
                validate_args)
            check_ops.assert_same_float_dtype([
                self._concentration1, self._concentration0])
            self._total_concentration = self._concentration1 + self._concentration0
        self.dtype = self._total_concentration.dtype,
        self.name = name
        graph_parents = [self._concentration1,
                         self._concentration0,
                         self._total_concentration]
        graph_parents = [] if graph_parents is None else graph_parents
        self._graph_parents = graph_parents

    @property
    def concentration1(self):
        """Concentration parameter associated with a `1` outcome."""
        return self._concentration1

    @property
    def concentration0(self):
        """Concentration parameter associated with a `0` outcome."""
        return self._concentration0

    @property
    def total_concentration(self):
        """Sum of concentration parameters."""
        return self._total_concentration

    def _maybe_assert_valid_concentration(self, concentration, validate_args):
        """Checks the validity of a concentration parameter."""
        if not validate_args:
            return concentration
        return control_flow_ops.with_dependencies([
            check_ops.assert_positive(
                concentration,
                message="Concentration parameter must be positive."),
        ], concentration)

    @property
    def batch_shape(self):
        """Shape of a single sample from a single event index as a `TensorShape`.
        May be partially defined or unknown.
        The batch dimensions are indexes into independent, non-identical
        parameterizations of this distribution.
        Returns:
          batch_shape: `TensorShape`, possibly unknown.
        """
        return tensor_shape.as_shape(self._batch_shape())

    def _batch_shape(self):
        return self.total_concentration.get_shape()

    def _expand_sample_shape_to_vector(self, x, name):
        """Helper to `sample` which ensures input is 1D."""
        x_static_val = tensor_util.constant_value(x)
        if x_static_val is None:
            prod = math_ops.reduce_prod(x)
        else:
            prod = np.prod(x_static_val, dtype=x.dtype.as_numpy_dtype())

        ndims = x.get_shape().ndims  # != sample_ndims
        if ndims is None:
            # Maybe expand_dims.
            ndims = array_ops.rank(x)
            expanded_shape = util.pick_vector(
                math_ops.equal(ndims, 0),
                np.array([1], dtype=np.int32), array_ops.shape(x))
            x = array_ops.reshape(x, expanded_shape)
        elif ndims == 0:
            # Definitely expand_dims.
            if x_static_val is not None:
                x = ops.convert_to_tensor(
                    np.array([x_static_val], dtype=x.dtype.as_numpy_dtype()),
                    name=name)
            else:
                x = array_ops.reshape(x, [1])
        elif ndims != 1:
            raise ValueError("Input is neither scalar nor vector.")

        return x, prod

    def _set_sample_static_shape(self, x, sample_shape):
        """Helper to `sample`; sets static shape info."""
        # Set shape hints.
        sample_shape = tensor_shape.TensorShape(
            tensor_util.constant_value(sample_shape))

        ndims = x.get_shape().ndims
        sample_ndims = sample_shape.ndims
        batch_ndims = self.batch_shape.ndims
        event_ndims = self.event_shape.ndims

        # Infer rank(x).
        if (ndims is None and
                sample_ndims is not None and
                batch_ndims is not None and
                event_ndims is not None):
            ndims = sample_ndims + batch_ndims + event_ndims
            x.set_shape([None] * ndims)

        # Infer sample shape.
        if ndims is not None and sample_ndims is not None:
            shape = sample_shape.concatenate([None] * (ndims - sample_ndims))
            x.set_shape(x.get_shape().merge_with(shape))

        # Infer event shape.
        if ndims is not None and event_ndims is not None:
            shape = tensor_shape.TensorShape(
                [None] * (ndims - event_ndims)).concatenate(self.event_shape)
            x.set_shape(x.get_shape().merge_with(shape))

        # Infer batch shape.
        if batch_ndims is not None:
            if ndims is not None:
                if sample_ndims is None and event_ndims is not None:
                    sample_ndims = ndims - batch_ndims - event_ndims
                elif event_ndims is None and sample_ndims is not None:
                    event_ndims = ndims - batch_ndims - sample_ndims
            if sample_ndims is not None and event_ndims is not None:
                shape = tensor_shape.TensorShape([None] * sample_ndims).concatenate(
                    self.batch_shape).concatenate([None] * event_ndims)
                x.set_shape(x.get_shape().merge_with(shape))

        return x

    @property

    def _event_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def event_shape(self):
        """Shape of a single sample from a single batch as a `TensorShape`.
        May be partially defined or unknown.
        Returns:
          event_shape: `TensorShape`, possibly unknown.
        """
        return tensor_shape.as_shape(self._event_shape())

    @contextlib.contextmanager
    def _name_scope(self, name=None, values=None):
        """Helper function to standardize op scope."""
        with ops.name_scope(self.name):
            with ops.name_scope(name, values=(
                    ([] if values is None else values) + self._graph_parents)) as scope:
                yield scope

    def _call_sample_n(self, sample_shape, seed, name, **kwargs):
        with self._name_scope(name, values=[sample_shape]):
            sample_shape = ops.convert_to_tensor(
                sample_shape, dtype=dtypes.int32, name="sample_shape")
            sample_shape, n = self._expand_sample_shape_to_vector(
                sample_shape, "sample_shape")
            samples = self._sample_n(n, seed, **kwargs)
            batch_event_shape = array_ops.shape(samples)[1:]
            final_shape = array_ops.concat([sample_shape, batch_event_shape], 0)
            samples = array_ops.reshape(samples, final_shape)
            samples = self._set_sample_static_shape(samples, sample_shape)
            return samples

    def _sample_n(self, n, seed=None):
        expanded_concentration1 = array_ops.ones_like(
            self.total_concentration, dtype=self.dtype) * self.concentration1
        expanded_concentration0 = array_ops.ones_like(
            self.total_concentration, dtype=self.dtype) * self.concentration0
        gamma1_sample = random_ops.random_gamma(
            shape=[n],
            alpha=expanded_concentration1,
            dtype=self.dtype,
            seed=seed)
        gamma2_sample = random_ops.random_gamma(
            shape=[n],
            alpha=expanded_concentration0,
            dtype=self.dtype,
            seed=distribution_util.gen_new_seed(seed, "beta"))
        beta_sample = gamma1_sample / (gamma1_sample + gamma2_sample)
        return beta_sample

    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _prob(self, x):
        return math_ops.exp(self._log_prob(x))

    def _log_cdf(self, x):
        return math_ops.log(self._cdf(x))

    def _cdf(self, x):
        return math_ops.betainc(self.concentration1, self.concentration0, x)

    def _log_unnormalized_prob(self, x):
        return (math_ops.xlogy(self.concentration1 - 1., x) +
                (self.concentration0 - 1.) * math_ops.log1p(-x))  # pylint: disable=invalid-unary-operand-type

    def _log_normalization(self):
        return (math_ops.lgamma(self.concentration1)
                + math_ops.lgamma(self.concentration0)
                - math_ops.lgamma(self.total_concentration))

    def _entropy(self):
        return (
                self._log_normalization()
                - (self.concentration1 - 1.) * math_ops.digamma(self.concentration1)
                - (self.concentration0 - 1.) * math_ops.digamma(self.concentration0)
                + ((self.total_concentration - 2.) *
                   math_ops.digamma(self.total_concentration)))

    def _mean(self):
        return self._concentration1 / self._total_concentration

    def _variance(self):
        return self._mean() * (1. - self._mean()) / (1. + self.total_concentration)

    def sample(self, sample_shape=(), seed=None, name="sample"):
        """Generate samples of the specified shape.
        Note that a call to `sample()` without arguments will generate a single
        sample.
        Args:
          sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
          seed: Python integer seed for RNG
          name: name to give to the op.
        Returns:
          samples: a `Tensor` with prepended dimensions `sample_shape`.
        """
        return self._call_sample_n(sample_shape, seed, name)
