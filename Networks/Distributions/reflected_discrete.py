import tensorflow as tf




class ReflectedProbabilityDistDiscrete(CategoricalProbabilityDistributionType):

    def __init__(self, size):
        super().__init__(size)

    def proba_distribution_from_latent(self, pi_latent_vector, pi_latent_vector_ref, vf_latent_vector, vf_latent_vector_ref, init_scale=1.0, init_bias=0.0):
        pdparam_1 = linear(pi_latent_vector, 'pi', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        pdparam_2 = linear(pi_latent_vector_ref, 'pi', self.n_cat, init_scale=init_scale, init_bias=init_bias)

        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = tf.split(pdparam_2, 10, 1)
        pdparam_2 = tf.concat([a0, a2, a1, a3, a5, a4, a6, a8, a7, a9], axis=1)
        pdparam = tf.divide(tf.add(pdparam_1, pdparam_2), 2)

        q_values = linear(vf_latent_vector, 'q', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

