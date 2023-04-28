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

def plot_hunting_initiation_vs_prey_in_field(model_name, assay_config, assay_id, n, figure_name):
    """Plots the hunting initiation rate against the number of prey in the visual field and practical visual range."""
    initiation_rate_compiled = []
    prey_in_visual_field_compiled = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")

        all_seq, all_ts = get_hunting_sequences_timestamps(data, False)
        all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)

        # Note can do the same for abort condition.
        try:
            initiation_rate = compute_initiation_rate_all_steps(len(all_steps), initiation_steps, window_size=200)
            prey_in_visual_field = get_prey_in_visual_field(data["fish_position"], data["fish_angle"], data["prey_positions"], 100)

            initiation_rate_compiled.append(initiation_rate)
            prey_in_visual_field_compiled.append(prey_in_visual_field)
        except IndexError:
            pass


    initiation_rate_compiled = np.concatenate(initiation_rate_compiled)
    prey_in_visual_field_compiled = np.concatenate(prey_in_visual_field_compiled)

    plt.scatter(initiation_rate_compiled, prey_in_visual_field_compiled)
    plt.ylabel("Prey in visual field")
    plt.xlabel("Hunting Initiation Rate")
    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey-initiation-rate-num-in-field-{figure_name}.jpg")
    plt.clf()
    plt.close()


def plot_initiation_rate_against_paramecium_density(model_name, assay_config, assay_id, n, figure_name):
    initiation_rate_compiled = []
    paramecium_density_compiled = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_seq, all_ts = get_hunting_sequences_timestamps(data, False)
        paramecium_density = np.array([get_paramecium_density(data["fish_position"][i], data["prey_positions"][i])
                              for i in range(data["fish_position"].shape[0])])
        all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)

        if len(initiation_steps) > 0:
            initiation_rate = compute_initiation_rate_all_steps(len(all_steps), initiation_steps, window_size=200)

            initiation_rate_compiled.append(initiation_rate)
            paramecium_density_compiled.append(paramecium_density)

    initiation_rate_compiled = np.concatenate(initiation_rate_compiled)
    paramecium_density_compiled = np.concatenate(paramecium_density_compiled)


    plt.scatter(initiation_rate_compiled, paramecium_density_compiled)
    plt.ylabel("Paramecium Density")
    plt.xlabel("Hunting Initiation Rate")
    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey-initiation-rate-density-{figure_name}.jpg")
    plt.clf()
    plt.close()


def show_conditions_vs_prey_density(model_name, assay_config, assay_id, n, figure_name):
    """Coloured scatter plot to indicate all prey density values and whether a hunt was initiated at the time"""

    all_steps_compiled = []
    hunting_steps_compiled = []
    initiation_steps_compiled = []
    abort_steps_compiled = []
    pre_capture_steps_compiled = []
    color_compiled = []
    paramecium_density_compiled = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")

        all_seq, all_ts = get_hunting_sequences_timestamps(data, False)

        paramecium_density = np.array([get_paramecium_density(data["fish_position"][i], data["prey_positions"][i])
                              for i in range(data["fish_position"].shape[0])])

        all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)

        color = np.zeros(all_steps.shape)
        if len(hunting_steps) > 0:
            color[hunting_steps] = 1
        if len(initiation_steps) > 0:
            color[initiation_steps] = 2
        if len(abort_steps) > 0:
            color[abort_steps] = 3
        if len(pre_capture_steps) > 0:
            color[pre_capture_steps] = 4

        all_steps_compiled.append(all_steps)
        hunting_steps_compiled.append(hunting_steps)
        initiation_steps_compiled.append(initiation_steps)
        abort_steps_compiled.append(abort_steps)
        pre_capture_steps_compiled.append(pre_capture_steps)
        color_compiled.append(color)
        paramecium_density_compiled.append(paramecium_density)

    # Time scatter plot
    for color, paramecium_density in zip(color_compiled, paramecium_density_compiled):
        plt.scatter([i for i in range(len(paramecium_density))], paramecium_density, c=color)
    plt.xlabel("Step")
    plt.ylabel("Paramecium Density")
    plt.savefig(f"../../../Analysis-Output/Behavioural/Time_scatter_vs_prey_density-{figure_name}.jpg")
    plt.clf()
    plt.close()

    # Convert all trial specific indices to concatenated indices.
    cumulative_length = 0
    for t, trial in enumerate(all_steps_compiled):
        all_steps_compiled[t] += cumulative_length
        hunting_steps_compiled[t] += cumulative_length
        initiation_steps_compiled[t] += cumulative_length
        abort_steps_compiled[t] += cumulative_length
        pre_capture_steps_compiled[t] += cumulative_length

        cumulative_length += len(trial)

    all_steps_compiled = np.concatenate(all_steps_compiled).astype(int)
    hunting_steps_compiled = np.concatenate(hunting_steps_compiled).astype(int)
    initiation_steps_compiled = np.concatenate(initiation_steps_compiled).astype(int)
    abort_steps_compiled = np.concatenate(abort_steps_compiled).astype(int)
    pre_capture_steps_compiled = np.concatenate(pre_capture_steps_compiled).astype(int)

    paramecium_density_compiled = np.concatenate(paramecium_density_compiled)

    boxes = []
    box_labels = []

    boxes.append(paramecium_density_compiled[all_steps_compiled])
    box_labels.append("All Steps")

    if len(hunting_steps_compiled) > 0:
        boxes.append(paramecium_density_compiled[hunting_steps_compiled])
        box_labels.append("Hunting")

    if len(initiation_steps_compiled) > 0:
        boxes.append(paramecium_density_compiled[initiation_steps_compiled])
        box_labels.append("Initiation")

    if len(abort_steps_compiled) > 0:
        boxes.append(paramecium_density_compiled[abort_steps_compiled])
        box_labels.append("Aborts")

    if len(pre_capture_steps_compiled) > 0:
        boxes.append(paramecium_density_compiled[pre_capture_steps_compiled])
        box_labels.append("Pre-Capture")

    # Conditional Boxplots
    fig, ax = plt.subplots()
    ax.boxplot(boxes)
    ax.set_xticklabels(box_labels)
    ax.set_ylabel("Paramecium Density")

    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey_density-boxplot-{figure_name}.jpg")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    # plot_hunting_initiation_vs_prey_in_field("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 20, "dqn_gamma-2")
    # plot_initiation_rate_against_paramecium_density("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 20, "dqn_gamma-2")
    show_conditions_vs_prey_density("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 100, "dqn_gamma-2")
