import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps
from Analysis.Behavioural.Hunting.get_hunting_conditions import get_hunting_conditions
from Analysis.Behavioural.Tools.get_fish_prey_incidence import get_fish_prey_incidence_multiple_prey

"""
Aim should be to find the parameters that determine success of hunting, hunting sequence initiation, etc. 
"""


# TOOLS

def get_paramecium_density(fish_position, prey_positions, max_distance=4250):
    """Computes a measure of nearby prey density - depends, nonlinearly on the distances of paramecia"""
    fish_prey_vectors = fish_position - prey_positions
    fish_prey_distances = (fish_prey_vectors[:, 0] ** 2 + fish_prey_vectors[:, 1] ** 2) ** 0.5
    max_used_distance = 300
    fish_prey_distances = fish_prey_distances[fish_prey_distances < max_used_distance]
    fish_prey_closeness = 1 - fish_prey_distances/max_distance
    fish_prey_closeness **= 5
    return np.sum(fish_prey_closeness)


def compute_initiation_rate_all_steps(num_steps, initiation_steps, window_size=100):
    initiation_rate = np.zeros((num_steps))
    initiation_count = np.zeros((num_steps))

    initiation_count[initiation_steps] += 1

    for w in range(0, num_steps):
        initiation_rate[w] = np.mean(initiation_count[w: min([w + window_size, num_steps])])

    return initiation_rate


def get_prey_in_visual_field(fish_position, fish_orientation, prey_positions, visual_distance, full_visual_field_angle=124.5):
    fish_prey_incidence = get_fish_prey_incidence_multiple_prey(fish_position, fish_orientation, prey_positions)
    within_visual_field = np.absolute(fish_prey_incidence) < full_visual_field_angle

    fish_prey_vectors = prey_positions - np.expand_dims(fish_position, 1)
    fish_prey_distance = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    within_visual_range = fish_prey_distance < visual_distance

    prey_in_visual_field = within_visual_field * within_visual_range * 1

    return np.sum(prey_in_visual_field, axis=1)


# PLOTS

def plot_hunting_initiation_vs_prey_in_field(data, figure_name):
    """Plots the hunting initiation rate against the number of prey in the visual field and practical visual range."""

    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)
    all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)

    # Note can do the same for abort condition.
    initiation_rate = compute_initiation_rate_all_steps(len(all_steps), initiation_steps, window_size=200)

    prey_in_visual_field = get_prey_in_visual_field(data["fish_position"], data["fish_angle"], data["prey_positions"], 100)

    plt.scatter(initiation_rate, prey_in_visual_field)
    plt.ylabel("Prey in visual field")
    plt.xlabel("Hunting Initiation Rate")
    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey-initiation-rate-num-in-field-{figure_name}.jpg")
    plt.clf()
    plt.close()


def plot_initiation_rate_against_paramecium_density(data, figure_name):
    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)
    paramecium_density = np.array([get_paramecium_density(data["fish_position"][i], data["prey_positions"][i])
                          for i in range(data["fish_position"].shape[0])])
    all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)

    initiation_rate = compute_initiation_rate_all_steps(len(all_steps), initiation_steps, window_size=200)

    plt.scatter(initiation_rate, paramecium_density)
    plt.ylabel("Paramecium Density")
    plt.xlabel("Hunting Initiation Rate")
    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey-initiation-rate-density-{figure_name}.jpg")
    plt.clf()
    plt.close()


def show_conditions_vs_prey_density(data, figure_name):
    """Coloured scatter plot to indicate all prey density values and whether a hunt was initiated at the time"""

    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)

    paramecium_density = np.array([get_paramecium_density(data["fish_position"][i], data["prey_positions"][i])
                          for i in range(data["fish_position"].shape[0])])

    all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)

    color = np.zeros(all_steps.shape)
    color[hunting_steps] = 1
    color[initiation_steps] = 2
    color[abort_steps] = 3
    color[pre_capture_steps] = 4

    # Time scatter plot
    plt.scatter([i for i in range(len(paramecium_density))], paramecium_density, c=color)
    plt.savefig(f"../../../Analysis-Output/Behavioural/Time_scatter_vs_prey_density-{figure_name}.jpg")
    plt.clf()
    plt.close()

    boxes = [paramecium_density[all_steps], paramecium_density[hunting_steps], paramecium_density[initiation_steps],
             paramecium_density[abort_steps], paramecium_density[pre_capture_steps]]

    # Conditional Boxplots
    fig, ax = plt.subplots()
    ax.boxplot(boxes)
    ax.set_xticklabels(["All Steps", "Hunting", "Initiation", "Aborts", "Pre-Capture"])

    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey_density-boxplot-{figure_name}.jpg")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    d1 = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    # plot_hunting_initiation_vs_prey_in_field(d1, "Test")
    plot_initiation_rate_against_paramecium_density(d1, "Test2")
    show_conditions_vs_prey_density(d1, "Test2")
