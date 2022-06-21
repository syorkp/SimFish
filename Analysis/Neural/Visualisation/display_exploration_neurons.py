
import numpy as np

import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.extract_exploration_sequences import extract_exploration_action_sequences_with_positions
from Analysis.load_stimuli_data import load_stimulus_data


def display_neurons_with_exploration_timestamps(model_name, assay_config, assay_id, neuron_indexes):
    data = load_data(f"dqn_scaffold_14-1", "Behavioural-Data-Free", f"Naturalistic-4")
    full_unit_activity = [[state[0, 0, j] for i, state in enumerate(data["rnn_state_actor"])] for j in range(512)]
    exploration_timestamps, exploration_sequences, exploration_fish_positions = \
        extract_exploration_action_sequences_with_positions(
        data)

    fig, ax = plt.subplots(figsize=(10, 5))

    for unit in neuron_indexes:
        unit_activity = full_unit_activity[unit]

        ax.plot(unit_activity/max(unit_activity))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    for i in exploration_timestamps:
        ax.hlines(-0.1, i[0], i[-1], color="g", linewidth=8)
    plt.xlabel("Step", fontsize=25)
    plt.ylabel("LSTM Unit Activity (Normalised)", fontsize=20)
    plt.tight_layout()
    plt.savefig("../../Figures/Panels/Panel-5/Exploration_unit.png")
    plt.show()


# display_neurons_with_exploration_timestamps("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic-4", [31, 45])
