import numpy as np


def get_paramecium_density(fish_position, prey_positions, max_distance=4250):
    """Computes a measure of nearby prey density - depends, nonlinearly on the distances of paramecia"""
    fish_prey_vectors = fish_position - prey_positions
    fish_prey_distances = (fish_prey_vectors[:, 0] ** 2 + fish_prey_vectors[:, 1] ** 2) ** 0.5
    max_used_distance = 300
    fish_prey_distances = fish_prey_distances[fish_prey_distances < max_used_distance]
    fish_prey_closeness = 1 - fish_prey_distances/max_distance
    fish_prey_closeness **= 5
    return np.sum(fish_prey_closeness)



