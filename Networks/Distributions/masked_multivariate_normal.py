import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
from tensorflow_probability.distributions import MultivariateNormalDiag


class MaskedMultivariateNormal(MultivariateNormalDiag):

    def __init__(self, mu_action, sigma_action):
        super().__init__(loc=mu_action, scale_diag=sigma_action)

        # Compute KDF here.
        mat = scipy.io.loadmat("bouts.mat")
        bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
        dist_angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
        distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
        distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

        distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

        impulse = (distance * 10 - (0.004644 * 140.0 + 0.081417)) / 1.771548
        dist_angles_radians = (np.absolute(dist_angles) / 180) * np.pi

        impulse = np.expand_dims(impulse, 1)
        dist_angles_radians = np.expand_dims(dist_angles_radians, 1)
        actions = np.concatenate((impulse, dist_angles_radians), axis=1)

        model = DBSCAN(eps=0.8125, min_samples=5).fit(actions)
        sorted_actions = actions[model.labels_ != -1]

        # Extra step - cut off negative impulse values
        sorted_actions = sorted_actions[sorted_actions[:, 0] >= 0]
        sorted_actions = sorted_actions[sorted_actions[:, 1] >= 0]

        self.kde_impulse = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 1], var_type='c', bw='cv_ml')
        self.kde_angle = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 0], var_type='c', bw='cv_ml')

    def sample(self, shape):
        val = MultivariateNormalDiag.sample(shape)

    def log_prob(self, action):
        probs = self.kde_impulse.pdf(action[:, 0]) * self.kde_angle.pdf(action[:, 1])
        probs[probs < 0.0002] = 0
        probs[probs > 0.0002] = 1

    def prob(self, action):
        raise NotImplementedError

    def entropy(self):
        ...
