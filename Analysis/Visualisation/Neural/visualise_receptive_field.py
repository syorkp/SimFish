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


def create_rnn_figure_overlaid(neuron_data1, neuron_data2, plot_number, n_subplots, n_prev_subplots, s_vector, trace_names):
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots/2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"RNN group {plot_number}", fontsize=16)

    for i in range(int(n_subplots/2)):
         axs[i, 0].plot(s_vector, neuron_data1[i], color="b", label=trace_names[0])
         axs[i, 0].plot(s_vector, neuron_data2[i], color="r", label=trace_names[1])
         axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} Response")
         axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots/2)):
         axs[i, 1].plot(s_vector, neuron_data1[i + int(n_subplots/2)], color="b", label=trace_names[0])
         axs[i, 1].plot(s_vector, neuron_data2[i + int(n_subplots/2)], color="r", label=trace_names[1])
         axs[i, 1].set_ylabel(f"Unit {i + int(n_subplots/2) + n_prev_subplots} Response")
         axs[i, 1].tick_params(labelsize=15)
    axs[0, 1].legend(bbox_to_anchor=(1, 2), loc='upper right', prop={'size': 24})

    axs[int(n_subplots/2-1), 0].set_xlabel("Stimulus Angle (pi radians)", fontsize=25)
    axs[int(n_subplots/2-1), 1].set_xlabel("Stimulus Angle (pi radians)", fontsize=25)
    fig.set_size_inches(18.5, 20)
    plt.show()


def plot_rnn_vectors_overlaid(s_vector, r_vector1, r_vector2, trace_names):
    n_subplots = len(r_vector1)
    n_per_plot = 16
    n_plots = math.ceil(n_subplots/n_per_plot)

    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data1 = r_vector1[(i * n_per_plot):]
            neuron_subset_data2 = r_vector2[(i * n_per_plot):]
        else:
            neuron_subset_data1 = r_vector1[(i * n_per_plot): (i * n_per_plot) + n_per_plot]
            neuron_subset_data2 = r_vector2[(i * n_per_plot): (i * n_per_plot) + n_per_plot]
        create_rnn_figure_overlaid(neuron_subset_data1, neuron_subset_data2, i + 1, len(neuron_subset_data1), i * n_per_plot, s_vector, trace_names)


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



data1 = load_data("even_prey_ref-5", "For-Traces", "Predator-Static-40")
data2 = load_data("even_prey_ref-5", "For-Traces", "Prey-Static-10")

stimulus_data1 = load_stimulus_data("even_prey_ref-5", "For-Traces", "Predator-Static-40")
stimulus_data2 = load_stimulus_data("even_prey_ref-5", "For-Traces", "Prey-Static-10")

stimulus_vector1 = get_stimulus_vector(stimulus_data1, "predator 1")
stimulus_vector2 = get_stimulus_vector(stimulus_data2, "prey 1")


all_vectors1 = get_all_neuron_vectors(data1, "predator 1", stimulus_data1, "rnn state")
all_vectors2 = get_all_neuron_vectors(data2, "prey 1", stimulus_data2, "rnn state")

# plot_rnn_vectors_overlaid(stimulus_vector1, all_vectors2, all_vectors1, ["Prey", "Predator"])

# plot_rnn_vectors(stimulus_vector1, all_vectors1)
#plot_rnn_vectors(stimulus_vector2, all_vectors2)
# plot_rnn_vectors(stimulus_vector3, all_vectors3)
