import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps
from Analysis.Behavioural.Hunting.get_hunting_conditions import get_hunting_conditions
from Analysis.Behavioural.Tools.get_fish_prey_incidence import get_fish_prey_incidence_multiple_prey


def plot_prey_density_across_visual_field_conditions(model_name, assay_config, assay_id, n):
    all_initiation_one_hot = []
    all_fish_prey_incidence = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_seq, all_ts = get_hunting_sequences_timestamps(data, False)
        all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)
        hunting_steps_one_hot = np.array([1 if i in hunting_steps else 0 for i in all_steps])
        initiation_steps_one_hot = np.array([1 if i in initiation_steps else 0 for i in all_steps])
        abort_steps_one_hot = np.array([1 if i in abort_steps else 0 for i in all_steps])
        pre_capture_steps_one_hot = np.array([1 if i in pre_capture_steps else 0 for i in all_steps])

        # Get fish prey incidence for all sequences
        prey_positions = data["prey_positions"]
        fish_positions = data["fish_position"]
        fish_orientation = data["fish_angle"]

        fish_prey_incidence = get_fish_prey_incidence_multiple_prey(fish_positions, fish_orientation, prey_positions)

        all_initiation_one_hot.append(initiation_steps_one_hot)
        all_fish_prey_incidence.append(fish_prey_incidence)

    all_initiation_one_hot = np.concatenate(all_initiation_one_hot)
    all_fish_prey_incidence = np.concatenate(all_fish_prey_incidence)

    all_relevant_incidence = all_fish_prey_incidence[all_initiation_one_hot]

    plt.hist(all_relevant_incidence.flatten())
    plt.show()


if __name__ == "__main__":
    plot_prey_density_across_visual_field_conditions("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic", 100)
