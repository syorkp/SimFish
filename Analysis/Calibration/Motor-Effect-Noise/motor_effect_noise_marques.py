import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


def remove_outliers(distances, angles, stim, bouts):
    impulse = np.expand_dims(distances, 1)
    dist_angles_radians = np.expand_dims(np.absolute(angles), 1)
    actions = np.concatenate((impulse, dist_angles_radians), axis=1)

    model = DBSCAN(eps=0.8125, min_samples=5).fit(actions)

    sorted_actions = actions[model.labels_ != -1]
    sorted_actions = sorted_actions[sorted_actions[:, 0] >= 0]
    sorted_actions = sorted_actions[sorted_actions[:, 1] >= 0]

    stim = stim[model.labels_ != -1]
    bouts = bouts[model.labels_ != -1]

    return sorted_actions, stim, bouts


def get_parameters_for_stimulation(actions, stimulations, chosen_stim, bouts):
    angles, distances = actions[:, 1], actions[:, 0]
    angles = angles[stimulations == chosen_stim]
    distances = distances[stimulations == chosen_stim]
    bouts = bouts[stimulations == chosen_stim]

    plt.scatter(np.abs(angles), distances, c=bouts)
    plt.show()


# Load Data
mat = scipy.io.loadmat("./Marques/bouts.mat")
bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
bout_inf_final_array = mat["BoutInfFinalArray"]

dist_angles = bout_kinematic_parameters_final_array[:, 10]  # This one
distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]
distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]
distances = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

stim = bout_inf_final_array[:, 6]
stim2 = bout_inf_final_array[:, 7]
bout = bout_inf_final_array[:, 133]


# Discount all those ignored in action mask.
actions, stim, bouts = remove_outliers(distances, dist_angles, stim, bout)

nums = []
for i in range(int(max(stim))):
    nums.append(len(actions[stim == i]))
    # get_parameters_for_stimulation(actions, stim, i, bouts)



