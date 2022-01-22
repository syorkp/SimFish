from tensorflow_probability.distributions import MultivariateNormalDiag


class MaskedMultivariateNormal(MultivariateNormalDiag):

    def __init__(self, mu_action, sigma_action):
        super().__init__(loc=mu_action, scale_diag=sigma_action)

    def sample(self, shape):
        val = MultivariateNormalDiag.sample(shape)


