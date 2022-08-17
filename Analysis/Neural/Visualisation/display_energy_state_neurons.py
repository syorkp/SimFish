import math
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from Analysis.load_data import load_data
from Analysis.Neural.Regression.label_cell_roles import get_category_indices
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron


def display_energy_state_neurons(model_name, assay_config, assay_id, neuron_indices):
    data = load_data(model_name, assay_config, assay_id)
    e = [i for i in data["energy_state"]][:3000]

    activity_profiles = [[state[0, 0, j] for i, state in enumerate(data["rnn_state_actor"])] for j in neuron_indices]

    fig, axs = plt.subplots(figsize=(10, 5))

    for neuron in activity_profiles:
        neuron_activity = normalise_within_neuron(neuron)
        # axs.plot((-np.array(neuron_activity ) /min(neuron_activity) ) +1, label="RNN Unit")
        axs.plot(np.absolute(neuron_activity[:3000]))

    consumption_points = [i for i, a in enumerate(data["consumed"]) if a == 1 and i < 3000]
    cs_points = [i for i, a in enumerate(data["action"]) if a == 3 and i < 3000]

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
    neurons = get_category_indices("dqn_scaffold_18-1", "Behavioural-Data-Endless", "Naturalistic", 3, "energy_state",
                                   score_threshold=0.9)

    display_energy_state_neurons("dqn_scaffold_18-1", "Behavioural-Data-Endless", "Naturalistic-1", neurons)
