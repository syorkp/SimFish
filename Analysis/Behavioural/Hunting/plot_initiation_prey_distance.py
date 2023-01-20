import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps, get_prey_being_hunted
from Analysis.Behavioural.Hunting.get_hunting_conditions import get_hunting_conditions

"""Create hist of prey distance at time of initiation."""


def plot_hist_initiation_prey_distance(model_name, assay_config, assay_id, n):
    prey_distances_being_hunted_initiation_compiled = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")

        all_seq, all_ts = get_hunting_sequences_timestamps(data, True)

        all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)

        if len(initiation_steps) > 0:
            hunted_prey = get_prey_being_hunted(data["prey_positions"], data["fish_position"], data["fish_angle"])
            prey_positions = data["prey_positions"]
            fish_position = data["fish_position"]
            fish_prey_vectors = prey_positions - np.expand_dims(fish_position, 1)

            prey_distances = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5

            prey_distances_being_hunted = prey_distances * (hunted_prey * 1)
            prey_distances_being_hunted_initiation = prey_distances_being_hunted[initiation_steps, :]
            prey_distances_being_hunted_initiation = prey_distances_being_hunted_initiation.flatten() / 10  # Convert to mm

            acceptable_range = (prey_distances_being_hunted_initiation > 0)
            prey_distances_being_hunted_initiation = prey_distances_being_hunted_initiation[acceptable_range]

            prey_distances_being_hunted_initiation_compiled.append(prey_distances_being_hunted_initiation)

    prey_distances_being_hunted_initiation_compiled = np.concatenate(prey_distances_being_hunted_initiation_compiled)

    plt.hist(prey_distances_being_hunted_initiation_compiled, bins=30)
    plt.xlabel("Distance (mm)")
    plt.title("Distance of Prey at Hunting Initiation")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    plot_hist_initiation_prey_distance("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 100)
