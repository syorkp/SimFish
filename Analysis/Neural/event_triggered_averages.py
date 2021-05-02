import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math


from Analysis.load_data import load_data
from Analysis.Behavioural.show_spatial_density import get_action_name


def get_event_triggered_average(data, event_name):
    indexes = [i for i, m in enumerate(data[event_name]) if m > 0]
    neuron_averages = [0 for i in range(len(data["rnn state"][0][0]))]
    neural_data = np.squeeze(data["rnn state"])
    neuron_baseline = [np.mean(data["rnn state"][:, :, i]) for i in range(len(data["rnn state"][0][0]))]
    for i in indexes:
        for j, n in enumerate(neuron_averages):
            neuron_averages[j] += neural_data[i][j]
    for s, n in enumerate(neuron_averages):
        neuron_averages[s] = 100 * ((n/len(indexes))-neuron_baseline[s])/neuron_baseline[s]
    return neuron_averages


def get_action_triggered_average(data):
    action_triggered_averages = {str(i): [0 for i in range(len(data["rnn state"][0][0]))] for i in range(10)}
    action_counts = {str(i): 0 for i in range(10)}
    neuron_baseline = [np.mean(data["rnn state"][:, :, i]) for i in range(len(data["rnn state"][0][0]))]
    for a, n in zip(data["behavioural choice"], np.squeeze(data["rnn state"])):
        for i, nn in enumerate(n):
            action_triggered_averages[str(a)][i] += nn
        action_counts[str(a)] += 1
    for a in action_counts.keys():
        if action_counts[a] > 2:
            for i, n in enumerate(action_triggered_averages[a]):
                action_triggered_averages[a][i] = (((n/action_counts[a]) - neuron_baseline[i])/neuron_baseline[i]) * 100
    return action_triggered_averages


def get_eta(data, event_name):
    if event_name == "actions":
        return get_action_triggered_average(data)
    else:
        return get_event_triggered_average(data, event_name)


def get_for_specific_neurons(atas, neuron_list):
    subset = {}
    for key in atas.keys():
        subset[key] = [atas[key][i] for i in neuron_list]
    return subset


# Timeseries versions

def get_ata_timeseries(data, window=5):
    ata_timeseries = {str(i): [[0 for j in range(2*window)] for k in range(len(data["rnn state"][0][0]))] for i in range(10)}
    action_counts = {str(i): 0 for i in range(10)}
    neuron_baseline = [np.mean(data["rnn state"][:, :, i]) for i in range(len(data["rnn state"][0][0]))]
    for i, a in enumerate(data["behavioural choice"]):
        if i < window or i > len(data["behavioural choice"])-window:
            continue
        for n in range(data["rnn state"].shape[-1]):
            ata_timeseries[str(a)][n] = ata_timeseries[str(a)][n] + data["rnn state"][i-window: i+window, 0, n]
        action_counts[str(a)] += 1
    for a in action_counts.keys():
        if action_counts[a] > 2:
            for i, n in enumerate(ata_timeseries[a]):
                ata_timeseries[str(a)][i] = (((n/action_counts[a]) - neuron_baseline[i])/neuron_baseline[i]) * 100
    return ata_timeseries


def get_eta_timeseries(data, event="consumed", window=5):
    eta_timeseries = [[0 for j in range(2*window)] for k in range(len(data["rnn state"][0][0]))]
    neuron_baseline = [np.mean(data["rnn state"][:, :, i]) for i in range(len(data["rnn state"][0][0]))]
    indexes = [i for i, m in enumerate(data[event]) if m > 0]
    for evi in indexes:
        for n, neuron in enumerate(eta_timeseries):
            eta_timeseries[n] = eta_timeseries[n] + data["rnn state"][evi - window: evi + window, 0, n]
    if len(indexes) > 1:
        for i, n in enumerate(eta_timeseries):
            eta_timeseries[i] = (((n / len(indexes)) - neuron_baseline[i]) / neuron_baseline[i]) * 100
    elif len(indexes) == 1:
        for i, n in enumerate(eta_timeseries):
            eta_timeseries[i] = ((n - neuron_baseline[i]) / neuron_baseline[i]) * 100
    return eta_timeseries


def get_predator_eta_timeseries(data, window=5):
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    if len(predator_timestamps) == 0:
        print("No predator data")
        return
    predator_sequence_timestamps = []
    current_sequence = []
    prev = 0
    while len(predator_timestamps) > 0:
        index = predator_timestamps.pop(0)
        if index == prev + 1 or prev == 0:
            current_sequence.append(index)
        else:
            predator_sequence_timestamps.append(current_sequence)
            current_sequence = [index]
        prev = index
    full_length = max([len(x) for x in predator_sequence_timestamps]) + window
    for i, sequence in enumerate(predator_sequence_timestamps):
        while len(sequence) < full_length:
            sequence.insert(0, sequence[0]-1)
        predator_sequence_timestamps[i] = sequence

    eta_timeseries = [[0 for j in range(full_length)] for k in range(len(data["rnn state"][0][0]))]
    neuron_baseline = [np.mean(data["rnn state"][:, :, i]) for i in range(len(data["rnn state"][0][0]))]
    for evi in predator_sequence_timestamps:
        for n, neuron in enumerate(eta_timeseries):
            eta_timeseries[n] = eta_timeseries[n] + data["rnn state"][evi[0]: evi[-1]+1, 0, n]
    if len(predator_sequence_timestamps) > 1:
        for i, n in enumerate(eta_timeseries):
            eta_timeseries[i] = (((n / len(predator_sequence_timestamps)) - neuron_baseline[i]) / neuron_baseline[i]) * 100
    elif len(predator_sequence_timestamps) == 1:
        for i, n in enumerate(eta_timeseries):
            eta_timeseries[i] = ((n - neuron_baseline[i]) / neuron_baseline[i]) * 100
    return eta_timeseries


def get_average_timeseries(timeseries_array, subset=None):
    if subset:
        timeseries_array = [timeseries_array[i] for i in subset]
    av = np.average(timeseries_array, axis=0)
    return av


data = load_data("even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic-1")


# ata = get_eta(data, "actions")
#
# ata_subset_1 = get_for_specific_neurons(ata, prey_only_4)
# ata_subset_2 = get_for_specific_neurons(ata, pred_only_4)
# plot_average_action_scores_comparison([ata, ata_subset_1, ata_subset_2], ["All Neurons", "Prey Only", "Predator Only"])
# plot_average_action_scores(ata_subset_1)
# plot_average_action_scores(ata_subset_2)
# boxplot_of_etas(ata_subset_1)
# boxplot_of_etas(ata_subset_2)
# shared_eta_distribution(ata_subset_1, ata_subset_2, ["prey only", "pred only"])
# eta_distribution(ata_subset_1, "Prey-Only")
# eta_distribution(ata_subset_2, "Predator-Only")
