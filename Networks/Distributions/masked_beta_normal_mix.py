import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
import tensorflow_probability as tfp

from Networks.Distributions.beta_normal_mix import BetaNormalDistribution


class MaskedBetaNormalDistribution(BetaNormalDistribution):

    def __init__(self, concentration1, concentration0, mu, std, impulse_scaling, angle_scaling):
        super().__init__(concentration1, concentration0, mu, std)

        # Compute KDF here.
        mat = scipy.io.loadmat("./Environment/Action_Space/Bout_classification/bouts.mat")
        bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
        dist_angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
        distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
        distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

        distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

        impulse = (distance * 10 - (0.004644 * 140.0 + 0.081417)) / 1.771548
        dist_angles_radians = (np.absolute(dist_angles) / 180) * np.pi

        impulse = impulse / impulse_scaling

        dist_angles_radians = dist_angles_radians / angle_scaling

        impulse = np.expand_dims(impulse, 1)
        dist_angles_radians = np.expand_dims(dist_angles_radians, 1)
        actions = np.concatenate((impulse, dist_angles_radians), axis=1)

        model = DBSCAN(eps=0.8125, min_samples=5).fit(actions)
        sorted_actions = actions[model.labels_ != -1]

        # Extra step - cut off negative impulse values
        sorted_actions = sorted_actions[sorted_actions[:, 0] >= 0]
        sorted_actions = sorted_actions[sorted_actions[:, 1] >= 0]

        # self.kde_impulse = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 0], var_type='c', bw='cv_ml')
        # self.kde_angle = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 1], var_type='c', bw='cv_ml')
        print("Creating action mask...")
        self.kde_impulse = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 0], var_type='c', bw='cv_ml')
        self.kde_angle = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 1], var_type='c', bw='cv_ml')

    def get_sample_masked_weights(self, actions, shape):
        probs = self.kde_impulse.pdf(actions[:, :, 0]) * self.kde_angle.pdf(np.absolute(actions[:, :, 1]))

        # Dis-allow negative impulses
        positive_impulses = ((actions[:, :, 0] >= 0) * 1)[:, 0]
        probs = probs * positive_impulses
        probs = np.nan_to_num(probs)

        # # Step function on probs
        probs[probs < 0.0000389489489] = 0.000001
        probs[probs >= 0.0000389489489] = 1

        integral = np.sum(probs)
        probs = probs/integral

        indices_chosen = np.random.choice(actions.shape[0], size=shape, p=probs, replace=False)
        actions_chosen = actions[indices_chosen, :, :]
        # return actions_chosen, probs, positive_impulses
        return actions_chosen

    def impose_zeroed(self, chosen_samples, mu_vals, threshold=0.01):
        """Brings values to exactly zero if they are below the threshold e.g. impulse of 0.001 will be returned as 0."""
        zero_chosen = ((mu_vals < threshold) * 1) * ((mu_vals > -threshold) * 1)
        zero_chosen = (zero_chosen == 0) * 1.0
        chosen_samples *= zero_chosen
        return chosen_samples

    def sample_masked(self, shape):
        preliminary_samples = self.sample(shape * 100)
        # self.preliminary_samples = preliminary_samples
        # chosen_samples, self.probs, self.positive_imp = tf.numpy_function(self.get_sample_masked_weights, [self.preliminary_samples, shape], [tf.float32, tf.float64, tf.int64])
        chosen_samples = tf.numpy_function(self.get_sample_masked_weights, [preliminary_samples, shape], tf.float32)
        mean_dist_beta = self.concentration1/(self.concentration0 + self.concentration1)
        mean_dist_vals = tf.concat([mean_dist_beta, self.mu], axis=1)
        chosen_samples_thresholded = tf.numpy_function(self.impose_zeroed, [chosen_samples, mean_dist_vals], tf.float32)
        return chosen_samples_thresholded
