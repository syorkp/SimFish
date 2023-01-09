import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps
from Analysis.Behavioural.Both.get_hunting_conditions import get_hunting_conditions

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


def show_conditions_vs_prey_density(data, figure_name):
    """Coloured scatter plot to indicate all prey density values and whether a hunt was initiated at the time"""

    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)

    paramecium_density = np.array([get_paramecium_density(data["fish_position"][i], data["prey_positions"][i])
                          for i in range(data["fish_position"].shape[0])])

    all_steps, hunting_steps, initiation_steps, abort_steps = get_hunting_conditions(data, all_ts)

    color = np.zeros(all_steps.shape)
    color[hunting_steps] = 1
    color[initiation_steps] = 2
    color[abort_steps] = 3

    # Time scatter plot
    plt.scatter([i for i in range(len(paramecium_density))], paramecium_density, c=color)
    plt.savefig(f"../../../Analysis-Output/Behavioural/Time_scatter_vs_prey_density-{figure_name}.jpg")
    plt.clf()
    plt.close()

    boxes = [paramecium_density[all_steps], paramecium_density[hunting_steps], paramecium_density[initiation_steps], paramecium_density[abort_steps]]
    # Conditional histograms
    fig, ax = plt.subplots()
    ax.boxplot(boxes)
    ax.set_xticklabels(["All Steps", "Hunting", "Initiation", "Aborts"])

    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey_density-boxplot-{figure_name}.jpg")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    d1 = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    show_conditions_vs_prey_density(d1, "Test")
