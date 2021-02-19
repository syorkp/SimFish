import numpy as np
import math
from matplotlib import pyplot as plt
from Analysis.load_data import load_data


def create_plot(neuron_data, plot_number, n_subplots, n_prev_subplots):
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots/2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"Neuron group {plot_number}", fontsize=16)

    for i in range(int(n_subplots/2)):
        axs[i, 0].plot(neuron_data[i])
        axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} activity")
        axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots/2)):
        axs[i, 1].plot(neuron_data[i + int(n_subplots/2)])
        axs[i, 1].set_ylabel(f"Unit {i + int(n_subplots/2) + n_prev_subplots} activity")
        axs[i, 1].tick_params(labelsize=15)

    axs[int(n_subplots/2-1), 0].set_xlabel("Step", fontsize=25)
    axs[int(n_subplots/2-1), 1].set_xlabel("Step", fontsize=25)
    fig.set_size_inches(18.5, 20)
    plt.show()


def plot_multiple_traces(neuron_data):
    n_subplots = len(neuron_data)
    n_per_plot = 30
    n_plots = math.ceil(n_subplots/n_per_plot)

    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data = neuron_data[(i * n_per_plot): ]
        else:
            neuron_subset_data = neuron_data[(i * n_per_plot): (i * n_per_plot) + n_per_plot]
        create_plot(neuron_subset_data, i + 1, len(neuron_subset_data), i * n_per_plot)



# data = load_data("large_all_features-1", "No_Stimuli", "No_Stimuli")
data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Curved_prey")


# rnn_unit_1 = [data["rnn state"][i-1][0][0] for i in data["step"][200:400]]
# rnn_unit_3 = [data["rnn state"][i-1][0][20] for i in data["step"][200:400]]
# rnn_unit_200 = [data["rnn state"][i-1][0][70] for i in data["step"][200:400]]
# conv_unit_1 = [data["rnn state"][i-1][0][69] for i in data["step"][200:400]]


unit_activity = [[data["rnn state"][i-1][0][j] for i in data["step"]] for j in range(512)]
plot_multiple_traces(unit_activity)

















