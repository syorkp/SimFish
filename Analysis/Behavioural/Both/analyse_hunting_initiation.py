import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps

"""
Aim should be to find the parameters that determine success of hunting, hunting sequence initiation, etc. 
"""


def get_paramecium_density(fish_position, prey_positions, max_distance=4250):
    """Computes a measure of nearby prey density - depends, nonlinearly on the distances of paramecia"""
    fish_prey_vectors = fish_position - prey_positions
    fish_prey_distances = (fish_prey_vectors[:, 0] ** 2 + fish_prey_vectors[:, 1] ** 2) ** 0.5
    max_used_distance = 300
    fish_prey_distances = fish_prey_distances[fish_prey_distances < max_used_distance]
    fish_prey_closeness = 1 - fish_prey_distances/max_distance
    fish_prey_closeness **= 5
    return np.sum(fish_prey_closeness)


def show_initiation_vs_prey_density(data):
    """Coloured scatter plot to indicate all prey density values and whether a hunt was initiated at the time"""
    # Get all steps for which there was hunting initiation
    # Get all steps for which there was exiting of a hunting sequence
    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)

    hunting_steps = list(set([a for at in all_ts for a in at]))

    paramecium_density = [get_paramecium_density(data["fish_position"][i], data["prey_positions"][i])
                          for i in range(data["fish_position"].shape[0])]
    color = [1 if i in hunting_steps else 0 for i, p in enumerate(paramecium_density)]

    plt.scatter([i for i in range(len(paramecium_density))], paramecium_density, c=color)
    plt.show()
    x = True

if __name__ == "__main__":
    d1 = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    show_initiation_vs_prey_density(d1)
