import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps, get_prey_being_hunted
from Analysis.Behavioural.Hunting.get_hunting_conditions import get_hunting_conditions
from Analysis.Behavioural.Hunting.compute_prey_velocity import compute_prey_velocity


def show_conditions_vs_prey_velocity(data, figure_name, fishcentric=False, use_only_hunted_prey=False):
    """Coloured scatter plot to indicate all prey density values and whether a hunt was initiated at the time"""

    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)

    paramecium_velocity, paramecium_speed = compute_prey_velocity(data["prey_positions"],
                                                                  fish_positions=data["fish_position"],
                                                                  egocentric=fishcentric)

    all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts,
                                                                                                        exclude_final_step=True)
    conditions = ["All Steps", "Hunting", "Initiation", "Aborts", "Pre-Capture"]

    color = np.zeros(all_steps.shape)
    color[hunting_steps] = 1
    color[initiation_steps] = 2
    color[abort_steps] = 3
    color[pre_capture_steps] = 4

    # Time scatter plot
    if use_only_hunted_prey:
        hunted_prey = get_prey_being_hunted(data["prey_positions"], data["fish_position"], data["fish_angle"])[:-1]
        prey_being_hunted = np.sum(hunted_prey, axis=1)
        prey_being_hunted_count = [[i for x in range(p)] for i, p in enumerate(prey_being_hunted)]
        prey_being_hunted_count = [p for pr in prey_being_hunted_count for p in pr]

        flattened_paramecium_velocity = paramecium_velocity[hunted_prey]
        color = color[prey_being_hunted_count]

        hunting_steps = np.array([1 if i in hunting_steps else 0 for i in range(len(all_steps))])
        initiation_steps = np.array([1 if i in initiation_steps else 0 for i in range(len(all_steps))])
        abort_steps = np.array([1 if i in abort_steps else 0 for i in range(len(all_steps))])
        pre_capture_steps = np.array([1 if i in pre_capture_steps else 0 for i in range(len(all_steps))])

        hunting_steps = np.expand_dims(hunting_steps, axis=1) * hunted_prey
        initiation_steps = np.expand_dims(initiation_steps, axis=1) * hunted_prey
        abort_steps = np.expand_dims(abort_steps, axis=1) * hunted_prey
        pre_capture_steps = np.expand_dims(pre_capture_steps, axis=1) * hunted_prey

        # Boxplots
        boxes = [paramecium_speed[all_steps].flatten(),
                 paramecium_speed[hunting_steps].flatten(),
                 paramecium_speed[initiation_steps].flatten(),
                 paramecium_speed[abort_steps].flatten(),
                 paramecium_speed[pre_capture_steps].flatten()]

    else:
        color = np.tile(color, paramecium_velocity.shape[1])
        flattened_paramecium_velocity = np.reshape(paramecium_velocity, (-1, 2))

        # Boxplots
        boxes = [paramecium_speed[all_steps].flatten(), paramecium_speed[hunting_steps].flatten(),
                 paramecium_speed[initiation_steps].flatten(),
                 paramecium_speed[abort_steps].flatten(), paramecium_speed[pre_capture_steps].flatten()]

    scatter = plt.scatter(flattened_paramecium_velocity[:, 0], flattened_paramecium_velocity[:, 1], c=color, alpha=0.4)
    plt.legend(handles=scatter.legend_elements()[0], labels=conditions)

    plt.savefig(f"../../../Analysis-Output/Behavioural/hunting_params_versus_prey_velocity-{figure_name}.jpg")
    plt.clf()
    plt.close()

    # Conditional Boxplots
    fig, ax = plt.subplots()
    ax.boxplot(boxes)
    ax.set_xticklabels(conditions)

    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey_speed-boxplot-{figure_name}.jpg")
    plt.clf()
    plt.close()



if __name__ == "__main__":
    d1 = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    show_conditions_vs_prey_velocity(d1, "Test", fishcentric=True, use_only_hunted_prey=True)




