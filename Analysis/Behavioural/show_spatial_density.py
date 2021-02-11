import numpy as np
import matplotlib.pyplot as plt
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
        nearby_predator_coordinates = data["predator_position"][step]

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


def get_relative_feature_density(feature_name, data):
    ...


data = load_data("changed_penalties-2", "Naturalistic", "Naturalistic-1")

prey_cloud = []

for i, step in enumerate(data["step"]):
    allocentric_prey, allocentric_predators = get_nearby_features(data, i)
    if len(allocentric_prey) > 0:
        egocentric_prey = transform_to_egocentric(allocentric_prey, data["position"][i], data["fish_angle"][i])
    else:
        egocentric_prey = []
    prey_cloud = prey_cloud + egocentric_prey

x, y = np.mgrid[slice(0, 800, 1), slice(0, 800, 1)]
locations = np.zeros((800, 800))

for i, j in prey_cloud:
    locations[round(i), round(j)] = 0.01

locations[300, 300] = 0.001
plt.pcolormesh(x, y, locations)
plt.show()

