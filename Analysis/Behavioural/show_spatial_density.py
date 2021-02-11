import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde


from Analysis.load_data import load_data

"""
To create a graph of the style in Figure 3b of Marques et al. (2018)
"""


def get_nearby_features(data, step, proximity=300):
    """For a single step, returns the positions of nearby prey and predators."""
    nearby_prey_coordinates = []
    nearby_predator_coordinates = []
    nearby_area = [[data["position"][step][0] - proximity,
                   data["position"][step][0] + proximity],
                   [data["position"][step][1] - proximity,
                    data["position"][step][1] + proximity]
                   ]

    for i in data["prey_positions"][step]:
        is_in_area = nearby_area[0][0] <= i[0] <= nearby_area[0][1] and \
                     nearby_area[1][0] <= i[1] <= nearby_area[1][1]
        if is_in_area:
            nearby_prey_coordinates.append(i)

    is_in_area = nearby_area[0][0] <= data["predator_position"][step][0] <= nearby_area[0][1] and \
                 nearby_area[1][0] <= data["predator_position"][step][1] <= nearby_area[1][1]
    if is_in_area:
        nearby_predator_coordinates = [data["predator_position"][step]]

    return nearby_prey_coordinates, nearby_predator_coordinates


def transform_to_egocentric(feature_positions, fish_position, fish_orientation):
    """Takes the feature coordinates and fish position and translates them onto an egocentric reference frame."""
    transformed_coordinates = []
    for i in feature_positions:
        v = [i[0]-fish_position[0], i[1]-fish_position[1]]
        theta = 2 * np.pi - fish_orientation
        tran_v = [np.cos(theta) * v[0]-np.sin(theta) * v[1],
                  np.sin(theta) * v[0]+np.cos(theta) * v[1]]
        tran_v[0] = tran_v[0] + 300
        tran_v[1] = tran_v[1] + 300
        transformed_coordinates.append(tran_v)
    return transformed_coordinates


def get_clouds_with_action(data, action=0):
    prey_cloud = []
    predator_cloud = []

    for i, step in enumerate(data["step"]):
        if data["behavioural choice"][i] == action:
            allocentric_prey, allocentric_predators = get_nearby_features(data, i)

            if len(allocentric_prey) > 0:
                egocentric_prey = transform_to_egocentric(allocentric_prey, data["position"][i], data["fish_angle"][i])
            else:
                egocentric_prey = []

            if len(allocentric_predators) > 0:
                egocentric_predators = transform_to_egocentric(allocentric_predators, data["position"][i], data["fish_angle"][i])
            else:
                egocentric_predators = []

            prey_cloud = prey_cloud + egocentric_prey
            predator_cloud = predator_cloud + egocentric_predators

    return prey_cloud, predator_cloud


def get_action_name(action_num):
    if action_num == 0:
        action_name = "Slow2"
    elif action_num == 1:
        action_name = "RT Right"
    elif action_num == 2:
        action_name = "RT Left"
    elif action_num == 3:
        action_name = "Capture Swim"
    elif action_num == 4:
        action_name = "j-turn Right"
    elif action_num == 5:
        action_name = "j-turn Left"
    elif action_num == 6:
        action_name = "Do Nothing"
    elif action_num == 7:
        action_name = "C-Start Right"
    elif action_num == 8:
        action_name = "C-Start Left"
    elif action_num == 9:
        action_name = "Approach Swim"
    else:
        action_name = "None"
    return action_name


def create_density_cloud(density_list, action_num, stimulus_name):
    x = np.array([i[0] for i in density_list])
    y = np.array([i[1] for i in density_list])
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.title(f"Feature: {stimulus_name}, Action: {get_action_name(action_num)}")
    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    plt.show()


data = load_data("changed_penalties-2", "Naturalistic", "Naturalistic-1")

for action_num in range(0, 10):
    prey_1, pred_1 = get_clouds_with_action(data, action_num)

    if len(prey_1) > 2:
        create_density_cloud(prey_1, action_num, "Prey")

    if len(pred_1) > 2:
        create_density_cloud(pred_1, action_num, "Predator")






