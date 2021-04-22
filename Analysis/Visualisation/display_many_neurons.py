import numpy as np
import math
from matplotlib import pyplot as plt
from Analysis.load_data import load_data
from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.Neural.calculate_vrv import get_stimulus_vector


def create_plot(neuron_data, plot_number, n_subplots, n_prev_subplots, stimulus_data=None):
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots / 2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"Neuron group {plot_number}", fontsize=16)

    for i in range(int(n_subplots / 2)):
        axs[i, 0].plot(neuron_data[i])
        if stimulus_data:
            for period in stimulus_data:
                for key in period.keys():
                    axs[i, 0].hlines(y=min(neuron_data[i]) - 1, xmin=period[key]["Onset"],
                                     xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                                     color="r")
        axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} activity")
        axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots / 2)):
        axs[i, 1].plot(neuron_data[i + int(n_subplots / 2)])
        if stimulus_data:
            for period in stimulus_data:
                for key in period.keys():
                    axs[i, 1].hlines(y=min(neuron_data[i + int(n_subplots / 2)]) - 1, xmin=period[key]["Onset"],
                                     xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                                     color="r")
        axs[i, 1].set_ylabel(f"Unit {i + int(n_subplots / 2) + n_prev_subplots} activity")
        axs[i, 1].tick_params(labelsize=15)

    # Add graph annotations
    axs[int(n_subplots / 2 - 1), 0].set_xlabel("Step", fontsize=25)

    axs[int(n_subplots / 2 - 1), 1].set_xlabel("Step", fontsize=25)

    fig.set_size_inches(18.5, 20)
    plt.show()


def create_plot_multiple_traces(neuron_data_traces, plot_number, n_subplots, n_prev_subplots, stimulus_data=None, trace_names=None):
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots / 2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"Neuron group {plot_number}", fontsize=16)

    for i in range(int(n_subplots / 2)):
        for j, trace in enumerate(neuron_data_traces):
            if trace_names:
                axs[i, 0].plot(trace[i], label=trace_names[j])
            else:
                axs[i, 0].plot(trace[i])
        if stimulus_data[0]:
            for period in stimulus_data[0]:
                for key in period.keys():
                    axs[i, 0].hlines(y=min(neuron_data_traces[0][i]) - 1,
                                     xmin=period[key]["Onset"],
                                     xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                                     color="r")
        axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} activity")
        axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots / 2)):
        for j, trace in enumerate(neuron_data_traces):
            if trace_names:
                axs[i, 1].plot(trace[i + int(n_subplots / 2)], label=trace_names[j])
            else:
                axs[i, 1].plot(trace[i + int(n_subplots / 2)])
        if stimulus_data[0]:
            for period in stimulus_data[0]:
                for key in period.keys():
                    axs[i, 1].hlines(y=min(neuron_data_traces[0][i + int(n_subplots / 2)]) - 1,
                                     xmin=period[key]["Onset"],
                                     xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                                     color="r")
        axs[i, 1].set_ylabel(f"Unit {i + int(n_subplots / 2) + n_prev_subplots} activity")
        axs[i, 1].tick_params(labelsize=15)
    if trace_names:
        axs[0, 1].legend(bbox_to_anchor=(1, 3), loc='upper right')

    # Add graph annotations
    axs[int(n_subplots / 2 - 1), 0].set_xlabel("Step", fontsize=25)

    axs[int(n_subplots / 2 - 1), 1].set_xlabel("Step", fontsize=25)

    fig.set_size_inches(18.5, 20)
    plt.savefig(f"Neuron group {plot_number}")
    plt.show()


def plot_traces(neuron_data, stimulus_data=None):
    n_subplots = len(neuron_data)
    n_per_plot = 30
    n_plots = math.ceil(n_subplots / n_per_plot)

    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data = neuron_data[(i * n_per_plot):]
        else:
            neuron_subset_data = neuron_data[(i * n_per_plot): (i * n_per_plot) + n_per_plot]
        create_plot(neuron_subset_data, i + 1, len(neuron_subset_data), i * n_per_plot, stimulus_data)


def plot_multiple_traces(neuron_data_list, stimulus_data_list=None, trace_names=None):
    n_subplots = len(neuron_data_list[0])
    n_per_plot = 30
    n_plots = math.ceil(n_subplots / n_per_plot)

    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data = [neuron_data[(i * n_per_plot):] for neuron_data in neuron_data_list]
        else:
            neuron_subset_data = [neuron_data[(i * n_per_plot): (i * n_per_plot) + n_per_plot] for neuron_data in
                                  neuron_data_list]
        create_plot_multiple_traces(neuron_subset_data, i + 1, len(neuron_subset_data[0]), i * n_per_plot,
                                    stimulus_data_list, trace_names=trace_names)


# data1 = load_data("even_prey_ref-5", "For-Traces", "Prey-Static-10")
# data2 = load_data("even_prey_ref-5", "For-Traces", "Predator-Static-40")
# stimulus_data1 = load_stimulus_data("even_prey_ref-5", "For-Traces", "Prey-Static-10")
# stimulus_data2 = load_stimulus_data("even_prey_ref-5", "For-Traces", "Predator-Static-40")
#
# unit_activity1 = [[data1["rnn state"][i - 1][0][j] for i in data1["step"]] for j in range(512)]
# unit_activity2 = [[data2["rnn state"][i - 1][0][j] for i in data2["step"]] for j in range(512)]
#
# plot_multiple_traces([unit_activity1, unit_activity2],
#                      [stimulus_data1, stimulus_data2],
#                      ["Prey", "Predator"])

# data1a = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-5")
# data2b = load_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Static-40")
# data1a = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Towards")
# data1b = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Away")
# data2a = load_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Towards")
# data2b = load_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Away")
#
# # stimulus_data1a = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-5")
# # stimulus_data2a = load_stimulus_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Static-40")
# stimulus_data1a = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Towards")
# stimulus_data1b = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Away")
# stimulus_data2a = load_stimulus_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Towards")
# stimulus_data2b = load_stimulus_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Away")


# unit_activity1a = [[data1a["rnn state"][i - 1][0][j] for i in data1a["step"]] for j in range(512)]
# unit_activity1b = [[data1b["rnn state"][i - 1][0][j] for i in data2b["step"]] for j in range(512)]
# unit_activity2a = [[data2a["rnn state"][i - 1][0][j] for i in data1a["step"]] for j in range(512)]
# unit_activity2b = [[data2b["rnn state"][i - 1][0][j] for i in data2b["step"]] for j in range(512)]

# Many presentations for RF mapping (Prey)
# plot_multiple_traces([unit_activity1a, unit_activity1b, unit_activity2a, unit_activity2b],
#                      [stimulus_data1a, stimulus_data1b, stimulus_data2a, stimulus_data2b],
#                      ["Prey-Towards", "Prey-Away", "Predator-Towards", "Predator-Away"])


# No vs red background
# plot_multiple_traces([unit_activity1a, unit_activity1],
#                      [stimulus_data1a, stimulus_data1],
#                      ["No Background", "Red Background"])

# Left vs right predators
# plot_multiple_traces([unit_activity2, unit_activity3],
#                      [stimulus_data2, stimulus_data3],
#                      ["Predator-Moving-Left", "Predator-Moving-Right"])

# Left vs right prey
# plot_multiple_traces([unit_activity5, unit_activity6],
#                      [stimulus_data5, stimulus_data6],
#                      ["Prey-Moving-Left", "Prey-Moving-Right"])

# # Static prey vs predators
# plot_multiple_traces([unit_activity1, unit_activity4],
#                      [stimulus_data1, stimulus_data4],
#                      ["Predator-Static", "Prey-Static"])
#
#
# Right prey vs static
# plot_multiple_traces([unit_activity4, unit_activity6],
#                      [stimulus_data4, stimulus_data6],
#                      ["Prey-Static", "Prey-Moving-Right"])

# # Prey of different sizes
# plot_multiple_traces([unit_activity7, unit_activity8, unit_activity9],
#                      [stimulus_data7, stimulus_data8, stimulus_data9],
#                      ["Prey-Size-5", "Prey-Size-10", "Prey-Size-15"])

# All
# plot_multiple_traces([unit_activity1, unit_activity2, unit_activity3, unit_activity4, unit_activity5, unit_activity6],
#                      [stimulus_data1, stimulus_data2, stimulus_data3, stimulus_data4, stimulus_data5, stimulus_data6],
#                      ["Predator-Static", "Predator-Moving-Left", "Predator-Moving-Right", "Prey-Static", "Prey-Moving-Left", "Prey-Moving-Right"])
