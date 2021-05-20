import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import seaborn as sns
import json

from Analysis.load_data import load_data
from Analysis.Neural.event_triggered_averages import get_eta, get_full_eta, get_for_specific_neurons, get_action_triggered_average, get_eta_timeseries, get_ata_timeseries, get_full_eta_timeseries, get_average_timeseries, get_predator_eta_timeseries, get_full_action_triggered_average, get_full_ata_timeseries
from Analysis.Neural.calculate_vrv import normalise_vrvs
from Analysis.Visualisation.Neural.visualise_response_vectors import order_vectors_by_kmeans
from Analysis.Behavioural.New.show_spatial_density import get_action_name


def display_all_atas(atas, groups=None):

    used_indexes = [int(key) for key in atas.keys() if sum(atas[key]) !=0]
    atas = [atas[key] for key in atas.keys() if sum(atas[key]) !=0]
    atas = [[atas[action][neuron] for action in range(len(atas))] for neuron in range(len(atas[0]))]
    for i, neuron in enumerate(atas):
        for j, action in enumerate(neuron):
            if action >1000:
                atas[i][j] = 1000
    atas = normalise_vrvs(atas)
    fig, ax = plt.subplots()
    fig.set_size_inches(1.85*len(used_indexes), 20)
    if groups:
        ordered_atas = []
        indexes = [j for sub in [groups[i] for i in groups.keys()] for j in sub]
        for i in indexes:
            ordered_atas.append(atas[i])
        ax.pcolor(ordered_atas, cmap='coolwarm')

    else:
        # atas, t, cat = order_vectors_by_kmeans(atas)
        ax.pcolor(atas, cmap='coolwarm')
    # ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    plt.xticks(range(len(used_indexes)), ["                    " + get_action_name(i) for i in used_indexes], fontsize=15)
    # ax.set_yticks(lat_grid, minor=True)

    plt.show()


def display_all_etas(etas, event_names, groups=None):
    for i, event in enumerate(etas):
        for j, neuron in enumerate(event):
            if neuron > 1000:
                etas[i][j] = 1000
    etas = [[etas[i][n] for i in range(len(etas))] for n in range(len(etas[0]))]
    etas = normalise_vrvs(etas)
    fig, ax = plt.subplots()
    fig.set_size_inches(1.85*len(event_names), 20)
    if groups:
        ordered_atas = []
        indexes = [j for sub in [groups[i] for i in groups.keys()] for j in sub]
        for i in indexes:
            ordered_atas.append(etas[i])
        ax.pcolor(ordered_atas, cmap='coolwarm')

    else:
        # atas, t, cat = order_vectors_by_kmeans(atas)
        ax.pcolor(etas, cmap='coolwarm')
    # ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    plt.xticks(range(len(event_names)), ["                    " + i for i in event_names], fontsize=15)
    # ax.set_yticks(lat_grid, minor=True)

    plt.show()


def shared_eta_distribution(event_triggered_averages_1, event_triggered_averages_2, labels):
    for a in range(10):
        atas = event_triggered_averages_1[str(a)]
        atas2 = event_triggered_averages_2[str(a)]
        if np.any(atas) or np.any(atas2):
            plt.figure(figsize=(5, 8))
            plt.hist([atas, atas2], color=["r", "b"], bins=range(-100, 100, 10))
            # sns.histplot(x=atas, color='skyblue', label='1', kde=True)
            # sns.histplot(x=atas2, color='red', label='2', kde=True)
            plt.title(f"ETAs for action {a}")
            # plt.xlim([-100, 100])
            plt.show()


def boxplot_of_etas(atas):
    labels = [key for key in atas.keys() if np.any(atas[key])]
    data = [atas[l] for l in labels]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=labels)
    ax.set_ylim([-1000, 1000])
    plt.show()


def neuron_action_scores(event_triggered_averages):
    for n in range(len(event_triggered_averages["0"])):
        neuron_responses = [event_triggered_averages[str(i)][n] for i in range(10)]
        plt.bar([get_action_name(i) for i in range(10)], neuron_responses)
        plt.show()


def plot_average_action_scores(event_triggered_averages):
    mean_scores = []
    for a in range(10):
        m = np.mean(event_triggered_averages[str(a)])
        mean_scores.append(m)
    plt.figure(figsize=(15, 5))
    plt.bar([get_action_name(i) for i in range(10)], mean_scores)
    plt.xlabel("Action-Triggered Average")
    plt.ylabel("Action")
    plt.show()


def plot_average_action_scores_comparison(atas, labels):
    atas2 = []
    used_actions = []
    for ata in atas:
        mean_scores = []
        for a in range(10):
            m = np.mean(ata[str(a)])
            if m != 0:
                mean_scores.append(m)
                used_actions.append(a)

        atas2.append(mean_scores)
    used_actions = list(set(used_actions))
    df = pd.DataFrame({label: data for data, label in zip(atas2, labels)})
    sns.set()
    df.plot.bar(rot=0, figsize=(10, 4.7))  # , color={labels[0]: "blue", labels[1]: "blue", labels[2]: "red"}
    plt.ylabel("Action-Triggered Average", fontsize=15)
    plt.xlabel("Bout", fontsize=15)
    plt.xticks([a for a in range(len(used_actions))], [get_action_name(a) for a in used_actions])
    plt.show()


def eta_distribution(event_triggered_averages, neuron_group="All Neurons"):
    for a in range(10):
        atas = event_triggered_averages[str(a)]
        if np.any(atas):
            plt.figure(figsize=(5, 8))
            sns.displot(atas)
            plt.title(f"{neuron_group}, ETAs for action {a}")
            plt.show()


def neuron_action_score_subplots(etas, plot_number, n_subplots, n_prev_subplots):
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots / 2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)
    fig.suptitle(f"Neuron group {plot_number}", fontsize=16)
    for i in range(int(n_subplots / 2)):
        axs[i, 0].bar([str(j) for j in range(10)], etas[i])
        axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} ETAs")
        axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots / 2)):
        axs[i, 1].bar([str(j) for j in range(10)], etas[(int(n_subplots / 2))+i])
        axs[i, 1].set_ylabel(f"Unit {i + n_prev_subplots + (int(n_subplots / 2))} ETAs")
        axs[i, 1].tick_params(labelsize=15)
    fig.set_size_inches(18.5, 20)
    plt.show()


def plot_all_action_scores(event_triggered_averages):
    n_subplots = len(event_triggered_averages["0"])
    n_per_plot = 30
    n_plots = math.ceil(n_subplots / n_per_plot)
    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data = [[event_triggered_averages[str(i)][n] for i in range(10)] for n in range(i * n_per_plot, n_subplots)]
        else:
            neuron_subset_data = [[event_triggered_averages[str(i)][n] for i in range(10)] for n in range(i * n_per_plot, (i*n_per_plot)+n_per_plot)]

        neuron_action_score_subplots(neuron_subset_data, i + 1, len(neuron_subset_data), i * n_per_plot)


def display_eta_timeseries(eta_timeseries, subset=None, title="No Title"):
    if subset:
        eta_timeseries = [eta_timeseries[i] for i in subset]
    for i, ts in enumerate(eta_timeseries):
        plt.plot(ts)
        plt.title(title + str(i))
        plt.axvline(x=(len(ts)/2), c="r")
        plt.show()


def display_average_eta_timeseries(eta_timeseries):
    sns.set()
    duration = len(eta_timeseries)
    plt.plot(range(-int(duration/2), int(duration/2), 1), eta_timeseries)
    plt.axvline(x=0, c="r")
    plt.show()


def display_multiple_average_eta_timeseries(eta_timeseries_list, group_labels):
    sns.set()
    duration = len(eta_timeseries_list[0])
    plt.figure(figsize=(8, 8))
    for label, eta_timeseries in zip(group_labels, eta_timeseries_list):
        plt.plot(range(-int(duration/2), int(duration/2)+1, 1), eta_timeseries, label=label)
    plt.axvline(x=0, c="r")
    plt.legend()
    plt.xlabel("Time to Prey Consumption (steps)")
    plt.ylabel("Normalised Event-Associated Activity")
    plt.show()


def display_ata_timeseries(ata_timeseries, subset=None):
    for a in ata_timeseries.keys():
        display_eta_timeseries(ata_timeseries[a], subset, f"Action: {a} ")


def display_eta_timeseries_overlay(timeseries_list):
    sns.set()
    for eta_time in timeseries_list:
        # plt.plot([np.log(e) if e > 0 else -np.log(e) for e in eta_time], alpha=0.3, color="r")
        plt.plot(np.log(eta_time), alpha=0.2, color="r")
    plt.axvline(x=0, c="r")
    plt.show()

# Single-point
# ata = get_full_action_triggered_average("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)
# ata = get_ata_timeseries(data)
with open(f"../../Categorisation-Data/even_prey_neuron_groups.json", 'r') as f:
    data2 = json.load(f)

# display_all_atas(ata)
#
# display_all_atas(ata, data2["new_even_prey_ref-4"])

# data = load_data("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic-1")
# ex1 = get_full_eta("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10, "exploration")
# ex2 = get_full_eta("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10, "consumed")
# ex3 = get_full_eta("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10, "predator")
# display_all_etas([ex1, ex2, ex3], ["exploration", "consumption", "predator"], data2["new_even_prey_ref-4"])

placeholder_list = data2["new_even_prey_ref-4"]["1"] + data2["new_even_prey_ref-4"]["8"] +\
                   data2["new_even_prey_ref-4"]["24"] + data2["new_even_prey_ref-4"]["29"]

# prey_in_front = get_for_specific_neurons(ata, placeholder_list)
# plot_average_action_scores_comparison([prey_in_front, ata], ["Prey-In-Front", "All Neurons"])


# Timeseries:

eta = get_full_eta_timeseries("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)
eta = np.absolute(eta)
prey_in_front_capture_average = get_average_timeseries(eta, placeholder_list)
other_capture_average = get_average_timeseries(eta, [i for i in range(512) if i not in placeholder_list])

display_multiple_average_eta_timeseries([prey_in_front_capture_average, other_capture_average], ["Prey in Front", "All Neurons"])

pred = get_full_eta_timeseries("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10, "predator")
av2 = get_average_timeseries(pred, placeholder_list)

# Overlay plots
# eta = normalise_vrvs(eta)
# eta = np.absolute(eta)
display_eta_timeseries_overlay([eta[i] for i in placeholder_list])# if -1000 < min(eta[i])  and max(eta[i]) < 1000])
display_eta_timeseries_overlay([e for i, e in enumerate(eta) if -1000 < min(e) and max(e) < 1000 and i not in placeholder_list])

display_average_eta_timeseries(av2)


# # 5
# prey_only_5 = [8, 13, 29, 44, 51, 52, 53, 59, 60, 65, 69, 76, 86, 90, 98, 116, 121, 122, 129, 134, 138, 142, 149, 157, 171, 173, 176, 182, 184, 188, 191, 201, 205, 217, 221, 232, 239, 250, 260, 289, 315, 328, 329, 332, 347, 362, 385, 390, 395, 399, 402, 406, 411, 415, 419, 429, 436, 446, 469, 481, 488, 497, 504]
# pred_only_5 = [5, 9, 15, 16, 17, 22, 41, 42, 46, 47, 56, 57, 70, 73, 77, 85, 89, 93, 95, 99, 100, 101, 106, 118, 120, 123, 127, 133, 136, 139, 145, 158, 163, 166, 172, 174, 178, 179, 181, 183, 190, 193, 194, 199, 200, 204, 208, 213, 220, 234, 236, 249, 251, 253, 255, 261, 263, 266, 269, 271, 277, 295, 296, 303, 307, 321, 338, 342, 364, 372, 381, 382, 383, 409, 424, 449, 450, 454, 456, 461, 462, 470, 474, 483, 495, 496, 501, 503, 505, 509, 511]
# prey_in_front_5 = [0, 6, 11, 21, 26, 27, 29, 31, 44, 48, 63, 71, 82, 87, 102, 105, 107, 109, 113, 115, 117, 121, 128, 137, 140, 141, 143, 144, 147, 153, 155, 159, 160, 165, 180, 182, 185, 197, 203, 205, 207, 209, 210, 215, 218, 227, 228, 240, 242, 243, 246, 247, 254, 268, 270, 273, 274, 278, 280, 283, 286, 287, 297, 298, 306, 308, 310, 311, 313, 322, 323, 324, 325, 332, 333, 334, 335, 341, 343, 344, 350, 351, 352, 354, 355, 361, 362, 366, 368, 374, 376, 397, 400, 402, 403, 412, 413, 416, 419, 423, 425, 426, 427, 428, 429, 430, 433, 435, 437, 439, 440, 441, 442, 446, 447, 448, 455, 459, 464, 466, 472, 473, 475, 476, 477, 482, 485, 486, 491, 493, 499, 500, 510]
#
#
# data = load_data("even_prey_ref-5", "Behavioural-Data-Free", "Naturalistic-1")
# # ata = get_ata_timeseries(data)
# # display_ata_timeseries(ata, [1, 2, 3])
#
# pred = get_predator_eta_timeseries(data)
# av = get_average_timeseries(pred, pred_only_5)
# display_average_eta_timeseries(av)
# av = get_average_timeseries(pred, prey_only_5)
# display_average_eta_timeseries(av)
# # display_eta_timeseries(pred, [1, 2, 3], "Predator ")
#
# eta = get_eta_timeseries(data)
# av = get_average_timeseries(eta, prey_only_5)
# display_average_eta_timeseries(av)
# av = get_average_timeseries(eta, pred_only_5)
# display_average_eta_timeseries(av)
# display_eta_timeseries(eta, [1, 2, 3], "Consumed ")


