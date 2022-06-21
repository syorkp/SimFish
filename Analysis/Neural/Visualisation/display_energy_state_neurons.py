import math
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from Analysis.load_data import load_data


def display_energy_state_neurons(model_name, assay_config, assay_id, neuron_indices):
    data = load_data(model_name, assay_config, assay_id)
    e = [i for i in data["energy_state"]]

    unit_activity1a = [[state[0, 0, j] for i, state in enumerate(data["rnn_state_actor"])] for j in range(512)]

    fig, axs = plt.subplots(figsize=(10, 5))

    for neuron in neuron_indices:
        neuron_activity = unit_activity1a[neuron]
        axs.plot((-np.array(neuron_activity ) /min(neuron_activity) ) +1, label="RNN Unit")

    consumption_points = [i for i, a in enumerate(data["consumed"]) if a == 1]
    cs_points = [i for i, a in enumerate(data["action"]) if a == 3]

    axs.plot((np.array(e)), label="Energy State")
    for c in cs_points:
        if c in consumption_points:
            plt.vlines(c, 1, 1.15, color="g", alpha=1, label="Successful Capture")
        else:
            plt.vlines(c, 1, 1.15, color="r", alpha=0.1, label="Capture Swim")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs.legend(by_label.values(), by_label.keys(), prop={'size': 13}, loc="upper right")
    plt.xlabel("Step", fontsize=25)
    plt.ylabel("Normalised Activity/Energy State", fontsize=18)
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    axs.tick_params(axis="x", labelsize=16)
    axs.tick_params(axis="y", labelsize=16)
    plt.tight_layout()
    plt.savefig("../../Figures/Panels/rnn_unit_energy_state.png")
    plt.show()


if __name__ == "__main__":
    display_energy_state_neurons("dqn_scaffold_18x-1", "Behavioural-Data-Free", "Naturalistic-1")