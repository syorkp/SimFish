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
            neuron_subset_data = neuron_data[(i * n_per_plot):]
        else:
            neuron_subset_data = neuron_data[(i * n_per_plot): (i * n_per_plot) + n_per_plot]
        create_plot(neuron_subset_data, i + 1, len(neuron_subset_data), i * n_per_plot)


data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Curved_prey")

n_c_neurons = data["left_conv_4"].shape[-1]

# a_neuron = [data["left_conv_4"][i, 0, :, 0] for i in range(len(data["left_conv_4"]))]

left_neurons = [[data["left_conv_4"][i, 0, :, j] for i in range(len(data["left_conv_4"]))] for j in range(n_c_neurons)]
right_neurons = [[data["right_conv_4"][i, 0, :, j] for i in range(len(data["right_conv_4"]))] for j in range(n_c_neurons)]

plot_multiple_traces(left_neurons)

x = True



