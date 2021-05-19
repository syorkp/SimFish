import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math


from Analysis.load_data import load_data
from Analysis.Behavioural.show_spatial_density import get_action_name


def get_full_action_triggered_average(model_name, configuration, assay_id, number_of_trials):
    action_triggered_averages = {str(i): [0 for i in range(512)] for i in range(10)}
    for i in range(1, number_of_trials+1):
        data = load_data(model_name, configuration, f"{assay_id}-{i}")
        new_ata = get_action_triggered_average(data)
        for key in new_ata.keys():
            action_triggered_averages[key] = [n+new_ata[key][i] for i, n in enumerate(action_triggered_averages[key])]
    for key in action_triggered_averages.keys():
        action_triggered_averages[key] = [action_triggered_averages[key][n]/number_of_trials for n in range(len(action_triggered_averages[key]))]
    return action_triggered_averages


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


def get_free_swimming_indexes(data):
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    wall_timestamps = [i for i, p in enumerate(data["position"]) if 200 < p[0] < 1300 and 200 < p[1] < 1300]
    prey_timestamps = []
    sensing_distance = 200
    for i, p in enumerate(data["position"]):
        for prey in data["prey_positions"][i]:
            sensing_area = [[p[0] - sensing_distance,
                             p[0] + sensing_distance],
                            [p[1] - sensing_distance,
                             p[1] + sensing_distance]]
            near_prey = sensing_area[0][0] <= prey[0] <= sensing_area[0][1] and \
                        sensing_area[1][0] <= prey[1] <= sensing_area[1][1]
            if near_prey:
                prey_timestamps.append(i)
                break
    # Check prey near at each step and add to timestamps.
    null_timestamps = predator_timestamps + wall_timestamps + prey_timestamps
    null_timestamps = set(null_timestamps)
    desired_timestamps = [i for i in range(len(data["behavioural choice"])) if i not in null_timestamps]
    return desired_timestamps


def get_exploration_triggered_average(data):
    indexes = get_free_swimming_indexes(data)
    neuron_averages = [0 for i in range(len(data["rnn state"][0][0]))]
    neural_data = np.squeeze(data["rnn state"])
    neuron_baseline = [np.mean(data["rnn state"][:, :, i]) for i in range(len(data["rnn state"][0][0]))]
    for i in indexes:
        for j, n in enumerate(neuron_averages):
            neuron_averages[j] += neural_data[i][j]
    for s, n in enumerate(neuron_averages):
        neuron_averages[s] = 100 * ((n/len(indexes))-neuron_baseline[s])/neuron_baseline[s]
    return neuron_averages


def get_eta(data, event_name):
    if event_name == "actions":
        return get_action_triggered_average(data)
    elif event_name == "exploration":
        return get_exploration_triggered_average(data)
    else:
        return get_event_triggered_average(data, event_name)


def get_for_specific_neurons(atas, neuron_list):
    subset = {}
    for key in atas.keys():
        subset[key] = [atas[key][i] for i in neuron_list]
    return subset


# Timeseries versions

def get_full_ata_timeseries(model_name, configuration, assay_id, number_of_trials, window=5):
    ata_timeseries = {str(i): [[0 for j in range(2*window)] for k in range(512)] for i in range(10)}
    for i in range(1, number_of_trials+1):
        data = load_data(model_name, configuration, f"{assay_id}-{i}")
        new_ata_t = get_ata_timeseries(data, window)
        for key in new_ata_t.keys():
            ata_timeseries[key] = [[m+new_ata_t[key][i][j] for j, m in enumerate(n)] for i, n in enumerate(ata_timeseries[key])]
    for key in ata_timeseries.keys():
        for i, neuron in enumerate(ata_timeseries[key]):
            ata_timeseries[key][i] = [neuron[n]/number_of_trials for n in range(len(neuron))]
    return ata_timeseries


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


def get_full_eta_timeseries(model_name, configuration, assay_id, number_of_trials, event="consumed", window=5):
    if event == "predator":
        return get_full_predator_eta_timeseries(model_name, configuration, assay_id, number_of_trials, window=5)
    eta_timeseries = [[0 for j in range(2*window)] for k in range(512)]
    for i in range(1, number_of_trials+1):
        data = load_data(model_name, configuration, f"{assay_id}-{i}")
        new_ata_t = get_eta_timeseries(data, event, window)
        eta_timeseries = [[m+new_ata_t[i][j] for j, m in enumerate(n)] for i, n in enumerate(eta_timeseries)]
    for i, neuron in enumerate(eta_timeseries):
        eta_timeseries[i] = [neuron[n]/number_of_trials for n in range(len(neuron))]
    return eta_timeseries


def get_eta_timeseries(data, event="consumed", window=5):
    eta_timeseries = [[0 for j in range(2*window)] for k in range(len(data["rnn state"][0][0]))]
    neuron_baseline = [np.mean(data["rnn state"][:, :, i]) for i in range(len(data["rnn state"][0][0]))]
    indexes = [i for i, m in enumerate(data[event]) if m > 0]
    for evi in indexes:
        for n, neuron in enumerate(eta_timeseries):
            if len(neuron) == window * 2 and len(data["rnn state"][evi - window: evi + window, 0, n])== 2*window:
                eta_timeseries[n] = eta_timeseries[n] + data["rnn state"][evi - window: evi + window, 0, n]
    if len(indexes) > 1:
        for i, n in enumerate(eta_timeseries):
            eta_timeseries[i] = (((n / len(indexes)) - neuron_baseline[i]) / neuron_baseline[i]) * 100
    elif len(indexes) == 1:
        for i, n in enumerate(eta_timeseries):
            eta_timeseries[i] = ((n - neuron_baseline[i]) / neuron_baseline[i]) * 100
    return eta_timeseries


def get_full_predator_eta_timeseries(model_name, configuration, assay_id, number_of_trials, window=5):
    eta_timeseries = [[0 for j in range(2 * window)] for k in range(512)]
    used_trials = 0
    for i in range(1, number_of_trials + 1):
        print(i)
        data = load_data(model_name, configuration, f"{assay_id}-{i}")
        new_ata_t = get_predator_eta_timeseries(data, window)
        if new_ata_t is not None:
            used_trials += 1
            eta_timeseries = [[m + new_ata_t[i][j] for j, m in enumerate(n)] for i, n in enumerate(eta_timeseries)]
    for i, neuron in enumerate(eta_timeseries):
        eta_timeseries[i] = [neuron[n] / used_trials for n in range(len(neuron))]
    return eta_timeseries


def get_predator_eta_timeseries(data, window=5):
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    if len(predator_timestamps) < 8:
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


def get_indexes_of_max(ex, total=100):
    ex = np.absolute(ex)
    ex = list(ex)
    hundred_most_associated = []
    while len(hundred_most_associated) < total:
        max_index = ex.index(max(ex))
        hundred_most_associated.append(max_index)
        ex[max_index] = 0
    plt.scatter([0 for i in range(512)], ex, alpha=0.2)
    plt.show()
    return hundred_most_associated


data = load_data("new_differential_prey_ref-4", "Behavioural-Data-Free-1", "Naturalistic-1")
ex = get_eta(data, "exploration")
ind = get_indexes_of_max(ex, 100)
print(ind)
x = True
# data = load_data("even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic-1")
# data = load_data("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic-1")
# ata = get_eta(data, "actions")
#
#
# with open(f"../Categorisation-Data/even_prey_neuron_groups.json", 'r') as f:
#     data2 = json.load(f)
#
# placeholder_list = data2["new_even_prey_ref-4"]["1"] + data2["new_even_prey_ref-4"]["8"] +\
#                    data2["new_even_prey_ref-4"]["24"] + data2["new_even_prey_ref-4"]["29"]
#
# ata_subset_1 = get_for_specific_neurons(ata, placeholder_list)

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
