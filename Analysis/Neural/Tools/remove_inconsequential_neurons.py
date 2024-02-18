import copy

import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Connectivity.load_network_variables import load_network_variables_dqn, load_network_variables_ppo
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces


def remove_those_with_no_output_advantage_only(rnn_data, model_name, conf_name, dqn=True, proportion_to_remove=0.1):
    # Load weights
    if dqn:
        all_weights = load_network_variables_dqn(model_name, conf_name, full_efference_copy=True)
    else:
        all_weights = load_network_variables_ppo(model_name, conf_name)

    # Extract relevant weights - those between RNN and output
    output_aw = all_weights["mainaw:0"]

    # Normalise activity profiles
    rnn_data_normalised = normalise_within_neuron_multiple_traces(rnn_data)
    rnn_n_units = rnn_data.shape[0]

    # Compute effect of each neuron on output layer at each timepoint.
    rnn_effect_aw = np.zeros((rnn_n_units, rnn_data.shape[1], output_aw.shape[1]))
    for i in range(10):
        rnn_effect_aw[:, :, i] = rnn_data_normalised * output_aw[:, i:i+1]

    rnn_effect_aw = np.sum(rnn_effect_aw, axis=(1, 2))

    # plt.scatter(rnn_effect_aw, [0 for i in range(256)], alpha=0.2)
    # plt.scatter(rnn_effect_vw, [0 for i in range(256)], alpha=0.2)
    # plt.show()

    rnn_effects = rnn_effect_aw.tolist()
    # plt.scatter(rnn_effects, [0 for i in range(512)])
    # plt.show()

    # Remove those for which this is essentially nothing.
    num_to_remove = int(proportion_to_remove * len(rnn_effects))
    indexes_to_remove = []
    rnn_effects = [abs(e) for e in rnn_effects]
    rnn_effects_cut = copy.copy(rnn_effects)
    for i in range(num_to_remove):
        min_value = min(rnn_effects_cut)
        rnn_effects_cut.remove(min_value)

        popped = rnn_effects.index(min_value)
        indexes_to_remove.append(popped)

    reduced_rnn_data = [rnn_d for i, rnn_d in enumerate(rnn_data) if i not in indexes_to_remove]
    return np.array(reduced_rnn_data)


def remove_those_with_no_output(rnn_data, model_name, conf_name, dqn=True, proportion_to_remove=0.1):
    # Load weights
    if dqn:
        all_weights = load_network_variables_dqn(model_name, conf_name, full_efference_copy=True)
    else:
        all_weights = load_network_variables_ppo(model_name, conf_name)

    # Extract relevant weights - those between RNN and output
    output_aw = all_weights["mainaw:0"]
    output_vw = all_weights["mainvw:0"]

    # Normalise activity profiles
    rnn_data_normalised = normalise_within_neuron_multiple_traces(rnn_data)
    half_rnn_n_units = int(rnn_data.shape[0]/2)

    # Compute effect of each neuron on output layer at each timepoint.
    rnn_effect_aw = np.zeros((half_rnn_n_units, rnn_data.shape[1], output_aw.shape[1]))
    for i in range(10):
        rnn_effect_aw[:, :, i] = rnn_data_normalised[:half_rnn_n_units] * output_aw[:, i:i+1]
    rnn_effect_vw = rnn_data_normalised[256:] * output_vw

    rnn_effect_aw = np.sum(rnn_effect_aw, axis=(1, 2))
    rnn_effect_vw = np.sum(rnn_effect_vw, axis=1)

    # plt.scatter(rnn_effect_aw, [0 for i in range(256)], alpha=0.2)
    # plt.scatter(rnn_effect_vw, [0 for i in range(256)], alpha=0.2)
    # plt.show()

    rnn_effects = np.concatenate((rnn_effect_aw, rnn_effect_vw)).tolist()
    # plt.scatter(rnn_effects, [0 for i in range(512)])
    # plt.show()

    # Remove those for which this is essentially nothing.
    num_to_remove = int(proportion_to_remove * len(rnn_effects))
    indexes_to_remove = []
    rnn_effects = [abs(e) for e in rnn_effects]
    rnn_effects_cut = copy.copy(rnn_effects)
    for i in range(num_to_remove):
        min_value = min(rnn_effects_cut)
        rnn_effects_cut.remove(min_value)

        popped = rnn_effects.index(min_value)
        indexes_to_remove.append(popped)

    reduced_rnn_data = [rnn_d for i, rnn_d in enumerate(rnn_data) if i not in indexes_to_remove]
    return np.array(reduced_rnn_data)


if __name__ == "__main__":
    data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", f"Naturalistic-1")
    rnn_data = np.swapaxes(data["rnn_state"][:, 0, 0, :], 0, 1)
    reduced_rnn_data = remove_those_with_no_output(rnn_data, "dqn_scaffold_18-1", "dqn_18_1")

