import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN
import scipy.stats as st
import statsmodels.api as sm
import tensorflow_probability as tfp
# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf


from Environment.Action_Space.Bout_classification.action_masking import produce_action_mask_version_3


class MaskedMultivariateNormal(tfp.distributions.MultivariateNormalDiag):

    def __init__(self, loc, scale_diag, impulse_scaling, angle_scaling):
        super().__init__(loc=loc, scale_diag=scale_diag, allow_nan_stats=False)

        self.mu_vals = loc
        print("Creating action mask...")
        self.kde, self.kdf_threshold = produce_action_mask_version_3()
        print("Action mask created")

    def get_sample_masked_weights(self, actions, shape):
        if actions.shape[1] > 1:
            print("Action mask problem")

        actions = actions[:, 0, :]
        actions = np.swapaxes(actions, 0, 1)

        probs = self.kde(actions)
        actions = np.swapaxes(actions, 0, 1)

        # Dis-allow negative impulses
        positive_impulses = ((actions[:, 0] >= 0) * 1)
        probs = probs * positive_impulses
        probs = np.nan_to_num(probs)

        # # Step function on probs
        probs[probs < self.kdf_threshold] = 0.000001
        probs[probs >= self.kdf_threshold] = 1

        integral = np.sum(probs)

        probs = probs/integral

        indices_chosen = np.random.choice(actions.shape[0], size=shape, p=probs, replace=False)
        actions_chosen = actions[indices_chosen, :]
        actions_chosen = np.expand_dims(actions_chosen, 1)

        return actions_chosen

    def impose_zeroed(self, chosen_samples, mu_vals, threshold=0.01):
        zero_chosen = ((mu_vals < threshold) * 1) * ((mu_vals > -threshold) * 1)
        zero_chosen = (zero_chosen == 0) * 1.0
        chosen_samples *= zero_chosen
        return chosen_samples

    def sample_masked(self, shape):
        preliminary_samples = self.sample(shape * 100)
        chosen_samples = tf.numpy_function(self.get_sample_masked_weights, [preliminary_samples, shape], tf.float32)
        chosen_samples_thresholded = tf.numpy_function(self.impose_zeroed, [chosen_samples, self.mu_vals], tf.float32)
        return chosen_samples_thresholded


