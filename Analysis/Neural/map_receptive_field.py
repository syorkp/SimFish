import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data
from Analysis.Neural.calculate_vrv import get_all_neuron_vectors, get_stimulus_vector, get_conv_neuron_vectors


def create_figure(neuron_data, plot_number, n_subplots, n_prev_subplots, s_vector):
    # TODO: Note, does not currently plot all channels, just one.
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots/2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"CNN group {plot_number}", fontsize=16)

    for i in range(int(n_subplots/2)):
        for j, channel in enumerate(neuron_data[i]):
            axs[i, 0].plot(s_vector, channel, label=j)
            axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} Response Vector")
            axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots/2)):
        for j, channel in enumerate(neuron_data[i + int(n_subplots/2)]):
            axs[i, 1].plot(s_vector, channel, label=j)
            axs[i, 1].set_ylabel(f"Unit {i + int(n_subplots/2) + n_prev_subplots} Response Vector")
            axs[i, 1].tick_params(labelsize=15)

    axs[int(n_subplots/2-1), 0].set_xlabel("Stimulus Angle", fontsize=25)
    axs[int(n_subplots/2-1), 1].set_xlabel("Stimulus Angle", fontsize=25)
    fig.set_size_inches(18.5, 20)
    plt.show()


def plot_conv_vectors(s_vector, r_vector):
    n_subplots = len(r_vector)
    n_per_plot = 30
    n_plots = math.ceil(n_subplots/n_per_plot)

    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data = r_vector[(i * n_per_plot):]
        else:
            neuron_subset_data = r_vector[(i * n_per_plot): (i * n_per_plot) + n_per_plot]
        create_figure(neuron_subset_data, i + 1, len(neuron_subset_data), i * n_per_plot, s_vector)


def create_rnn_figure(neuron_data, plot_number, n_subplots, n_prev_subplots, s_vector):
    # TODO: Note, does not currently plot all channels, just one.
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots/2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"RNN group {plot_number}", fontsize=16)

    for i in range(int(n_subplots/2)):
         axs[i, 0].plot(s_vector, neuron_data[i])
         axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} Response Vector")
         axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots/2)):
         axs[i, 1].plot(s_vector, neuron_data[i + int(n_subplots/2)])
         axs[i, 1].set_ylabel(f"Unit {i + int(n_subplots/2) + n_prev_subplots} Response Vector")
         axs[i, 1].tick_params(labelsize=15)

    axs[int(n_subplots/2-1), 0].set_xlabel("Stimulus Angle", fontsize=25)
    axs[int(n_subplots/2-1), 1].set_xlabel("Stimulus Angle", fontsize=25)
    fig.set_size_inches(18.5, 20)
    plt.show()


def plot_rnn_vectors(s_vector, r_vector):
    n_subplots = len(r_vector)
    n_per_plot = 30
    n_plots = math.ceil(n_subplots/n_per_plot)

    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data = r_vector[(i * n_per_plot):]
        else:
            neuron_subset_data = r_vector[(i * n_per_plot): (i * n_per_plot) + n_per_plot]
        create_rnn_figure(neuron_subset_data, i + 1, len(neuron_subset_data), i * n_per_plot, s_vector)


# data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Prey-Moving-Left")
# stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli", "Prey-Moving-Left")
# stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
# all_vectors = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn state")
# plot_rnn_vectors(stimulus_vector, all_vectors)
#
#
# data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Prey-Moving-Right")
# stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli",  "Prey-Moving-Right")
# stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
# all_vectors = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn state")
# plot_rnn_vectors(stimulus_vector, all_vectors)
#
#
# data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Prey-Static")
# stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli",  "Prey-Static")
# stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
# all_vectors = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn state")
# plot_rnn_vectors(stimulus_vector, all_vectors)

# For Prey
data1 = load_data("even_prey_ref-4", "Receptive_Field_Mapping", "Predator-Static")
data2 = load_data("even_prey_ref-4", "Receptive_Field_Mapping", "Predator-Moving-Left")
data3 = load_data("even_prey_ref-4", "Receptive_Field_Mapping", "Predator-Moving-Right")

stimulus_data1 = load_stimulus_data("even_prey_ref-4", "Receptive_Field_Mapping", "Predator-Static")
stimulus_data2 = load_stimulus_data("even_prey_ref-4", "Receptive_Field_Mapping", "Predator-Moving-Left")
stimulus_data3 = load_stimulus_data("even_prey_ref-4", "Receptive_Field_Mapping", "Predator-Moving-Right")

stimulus_vector1 = get_stimulus_vector(stimulus_data1, "predator 1")
stimulus_vector2 = get_stimulus_vector(stimulus_data2, "predator 1")
stimulus_vector3 = get_stimulus_vector(stimulus_data3, "predator 1")


all_vectors1 = get_all_neuron_vectors(data1, "predator 1", stimulus_data1, "rnn state")
all_vectors2 = get_all_neuron_vectors(data2, "predator 1", stimulus_data2, "rnn state")
all_vectors3 = get_all_neuron_vectors(data3, "predator 1", stimulus_data3, "rnn state")


plot_rnn_vectors(stimulus_vector1, all_vectors1)
#plot_rnn_vectors(stimulus_vector2, all_vectors2)
# plot_rnn_vectors(stimulus_vector3, all_vectors3)
