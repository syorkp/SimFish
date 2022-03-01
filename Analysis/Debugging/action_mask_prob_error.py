import scipy.io
import numpy as np
from sklearn.cluster import DBSCAN
import statsmodels.api as sm


class MaskedMultivariateNormal():

    def __init__(self, impulse_scaling, angle_scaling):
        # Compute KDF here.
        mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")
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


x = MaskedMultivariateNormal(10, np.pi / 5)