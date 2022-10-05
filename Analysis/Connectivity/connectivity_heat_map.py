import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces
from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.rnn_interconnectivity import get_rnn_interconnectivity


def display_connectivity_heat_map(connectivity_matrix, name):
    n_neurons = connectivity_matrix.shape[1]
    connectivity_matrix_1 = connectivity_matrix[:n_neurons]
    connectivity_matrix_2 = connectivity_matrix[n_neurons:]

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].pcolormesh(connectivity_matrix_1, cmap="bwr")
    axs[1].pcolormesh(connectivity_matrix_2, cmap="bwr")
    plt.savefig(f"connectivity_map-{name}.jpg", dpi=100)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    network_variables_1 = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", True)
    rnn_interconnectivity_1 = get_rnn_interconnectivity(network_variables_1, gate_num=0)
    display_connectivity_heat_map(rnn_interconnectivity_1["main_rnn/lstm_cell/kernel:0"], "gate-0")

    rnn_interconnectivity_1 = get_rnn_interconnectivity(network_variables_1, gate_num=1)
    display_connectivity_heat_map(rnn_interconnectivity_1["main_rnn/lstm_cell/kernel:0"], "gate-1")

    rnn_interconnectivity_1 = get_rnn_interconnectivity(network_variables_1, gate_num=2)
    display_connectivity_heat_map(rnn_interconnectivity_1["main_rnn/lstm_cell/kernel:0"], "gate-2")

    rnn_interconnectivity_1 = get_rnn_interconnectivity(network_variables_1, gate_num=3)
    display_connectivity_heat_map(rnn_interconnectivity_1["main_rnn/lstm_cell/kernel:0"], "gate-3")

