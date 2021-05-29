import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import seaborn as sns
import json

from Analysis.Neural.New.event_triggered_averages import get_full_eta_timeseries, get_ata_timeseries, get_for_specific_neurons, get_full_eta, get_average_timeseries, get_full_action_triggered_average
from Analysis.Neural.New.calculate_vrv import normalise_vrvs
from Analysis.Behavioural.New.show_spatial_density import get_action_name
from Analysis.load_data import load_data


def display_all_atas(atas, groups=None):

    used_indexes = [int(key) for key in atas.keys() if sum(atas[key]) !=0]
    atas = [atas[key] for key in atas.keys() if sum(atas[key]) !=0]
    atas = [[atas[action][neuron] for action in range(len(atas))] for neuron in range(len(atas[0]))]
    for i, neuron in enumerate(atas):
        for j, action in enumerate(neuron):
            if action > 1000:
                atas[i][j] = 1000
            elif action < -1000:
                atas[i][j] = -1000

    atas = normalise_vrvs(atas)
    fig, ax = plt.subplots()
    fig.set_size_inches(1.85*len(used_indexes), 20)
    ax.tick_params(labelsize=15)
    if groups:
        ordered_atas = []
        indexes = [j for sub in [groups[i] for i in groups.keys()] for j in sub]
        for i in indexes:
            ordered_atas.append(atas[i])
        ax.pcolor(ordered_atas, cmap='coolwarm')

        transition_points = [len(groups[key]) for i, key in enumerate(groups.keys())]
        cumulative_tps = []
        for i, t in enumerate(transition_points):
            if i == 0:
                continue
            cumulative_tps.append(sum(transition_points[:i]))
        transition_points = cumulative_tps

        # cluster_labels = [i for i in range(len(transition_points))]

        def format_func_cluster(value, tick):
            for i, tp in enumerate(transition_points):
                if value < tp:
                    return i
            return len(transition_points)
        for t in transition_points:
            ax.axhline(t, color="black", linewidth=1)
        ax.set_yticks(transition_points, minor=True)
        ax2 = ax.secondary_yaxis("right")
        ax2.tick_params(axis='y', labelsize=20)
        # ax2.yaxis.grid(True, which='minor', linewidth=20, linestyle='-', color="b")
        ax2.yaxis.set_major_locator(plt.FixedLocator(transition_points))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func_cluster))
        ax2.set_ylabel("Cluster ID", fontsize=40)
        ax2.tick_params(axis='y', labelsize=20)
    else:
        # atas, t, cat = order_vectors_by_kmeans(atas)
        ax.pcolor(atas, cmap='coolwarm')
    # ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    plt.xticks(range(len(used_indexes)), [get_action_name(i) for i in used_indexes], fontsize=25)
    ax.set_ylabel("Neuron", fontsize=40)
    ax.set_xlabel("Bout Choice", fontsize=40)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()

    plt.show()


def display_all_etas(etas, event_names, groups=None):
    for i, event in enumerate(etas):
        for j, neuron in enumerate(event):
            if neuron > 1000:
                etas[i][j] = 1000
            elif neuron < -1000:
                etas[i][j] = -1000
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

        transition_points = [len(groups[key]) for i, key in enumerate(groups.keys())]
        cumulative_tps = []
        for i, t in enumerate(transition_points):
            if i == 0:
                continue
            cumulative_tps.append(sum(transition_points[:i]))
        transition_points = cumulative_tps

        # cluster_labels = [i for i in range(len(transition_points))]

        def format_func_cluster(value, tick):
            for i, tp in enumerate(transition_points):
                if value < tp:
                    return i
            return len(transition_points)
        for t in transition_points:
            ax.axhline(t, color="black", linewidth=1)
        ax.set_yticks(transition_points, minor=True)
        ax2 = ax.secondary_yaxis("right")
        # ax2.yaxis.grid(True, which='minor', linewidth=20, linestyle='-', color="b")
        ax2.yaxis.set_major_locator(plt.FixedLocator(transition_points))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func_cluster))
        ax2.set_ylabel("Cluster ID", fontsize=40)
        ax2.tick_params(axis='y', labelsize=20)
    else:
        # atas, t, cat = order_vectors_by_kmeans(atas)
        ax.pcolor(etas, cmap='coolwarm')
    # ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    plt.xticks(range(len(event_names)), [i for i in event_names], fontsize=25)
    # ax.set_yticks(lat_grid, minor=True)
    ax.tick_params(axis="x", labelrotation=45)

    ax.set_ylabel("Neuron", fontsize=20)
    ax.set_xlabel("Event", fontsize=40)
    fig.tight_layout()
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


def plot_average_action_scores_comparison(atas, labels, stds=None):
    atas2 = []
    used_actions = []
    for ata in atas:
        mean_scores = []
        for a in range(10):
            m = np.mean(ata[str(a)])
            if m != 0:
                mean_scores.append(m*0.35)
                used_actions.append(a)
        atas2.append(mean_scores)
    atas2 = normalise_vrvs(atas2)
    used_actions = list(set(used_actions))
    df = pd.DataFrame({label: data for data, label in zip(atas2, labels)})
    a = [[0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]]
    sns.set()
    df.plot.bar(rot=0, figsize=(10, 6), yerr=stds)  # , color={labels[0]: "blue", labels[1]: "blue", labels[2]: "red"}
    plt.ylabel("Action-Triggered Average", fontsize=20)
    plt.xlabel("Bout", fontsize=20)
    plt.xticks([a for a in range(len(used_actions))], [get_action_name(a) for a in used_actions], fontsize=15)
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


def display_multiple_average_eta_timeseries(eta_timeseries_list, group_labels, std=None, std2=None, std3=None):
    sns.set()
    duration = len(eta_timeseries_list[0])
    plt.figure(figsize=(8, 8))
    for label, eta_timeseries in zip(group_labels, eta_timeseries_list):
        plt.plot(range(-int(duration/2), int(duration/2)+1, 1), eta_timeseries, label=label)
    plt.axvline(x=0, c="r")
    plt.yticks(np.linspace(0, 0.13, 11), [i/10 for i in range(0, 11)])
    plt.fill_between(range(-10, 11), [eta_timeseries_list[0][i]-stdi for i, stdi in enumerate(std)], [eta_timeseries_list[0][i]+stdi for i, stdi in enumerate(std)], alpha=0.4)
    plt.fill_between(range(-10, 11), [eta_timeseries_list[2][i]-stdi for i, stdi in enumerate(std2)], [eta_timeseries_list[2][i]+stdi for i, stdi in enumerate(std2)], alpha=0.4, color="g")
    plt.fill_between(range(-10, 11), [eta_timeseries_list[3][i]-stdi for i, stdi in enumerate(std3)], [eta_timeseries_list[3][i]+stdi for i, stdi in enumerate(std3)], alpha=0.4, color="r")
    plt.legend(fontsize=15)
    plt.xlabel("Time to Prey Consumption (steps)", fontsize=20)
    plt.ylabel("Normalised EAA",  fontsize=20)
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

import scipy.stats as stats


def check_separation(group1, group2):
    plt.figure()
    plt.hist(group1, alpha=0.5, bins=100)
    plt.hist(group2, alpha=0.5, bins=100)
    plt.show()
    print(stats.f_oneway(group1, group2))


# with open(f"../../Categorisation-Data/latest_even.json", 'r') as f:
#     data2 = json.load(f)

#
with open(f"../../Categorisation-Data/final_even2.json", 'r') as f:
    data2 = json.load(f)

# Single-point
ata = get_full_action_triggered_average("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)

# display_all_atas(ata, data2["new_even_prey_ref-4"])

data = load_data("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic-1")
ex1 = get_full_eta("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10, "exploration")
ex2 = get_full_eta("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10, "consumed")
ex3 = get_full_eta("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10, "predator")

# display_all_etas([ex1, ex2, ex3], ["Exploration", "Consumption", "Predator"], data2["new_even_prey_ref-4"])


placeholder_list = data2["new_even_prey_ref-4"]["3"] + data2["new_even_prey_ref-4"]["16"]
predator_only_ns = [2, 13, 19, 23, 27, 29, 34, 36, 46, 50, 60, 66, 81, 82, 93, 94, 95, 99, 100, 106, 110, 113, 117, 119, 122, 135, 145, 150, 156, 163, 165, 169, 171, 174, 182, 185, 186, 201, 203, 217, 218, 219, 220, 225, 226, 227, 238, 244, 259, 261, 264, 269, 280, 290, 302, 308, 310, 317, 322, 324, 339, 341, 345, 350, 366, 373, 402, 411, 450, 464, 469, 471, 477, 493]
prey_only_ns = [72, 77, 82, 138, 222, 232, 253, 268, 279, 318, 369, 382, 385, 388, 410, 433, 461, 481]
predator_cs_ns = data2["new_even_prey_ref-4"]["15"] + data2["new_even_prey_ref-4"]["11"]
valence_ns = data2["new_even_prey_ref-4"]["3"] + data2["new_even_prey_ref-4"]["9"]
prey_full_field_ns = data2["new_even_prey_ref-4"]["7"] + data2["new_even_prey_ref-4"]["8"] + data2["new_even_prey_ref-4"]["9"]+ data2["new_even_prey_ref-4"]["12"]

ex2 = normalise_vrvs([ex1, ex2, ex3])[1]
atas_v = [ata[i] for i in ata.keys()]
atas_v = normalise_vrvs(atas_v)
prey_in_front_etas_consumption = [a for i, a in enumerate(ex2) if i in placeholder_list]
prey_in_front_etas_scs = [a for i, a in enumerate(atas_v[3]) if i in placeholder_list]
all_etas_consumption = [a for i, a in enumerate(ex2) if i not in placeholder_list]
all_etas_scs = [a for i, a in enumerate(ata["3"]) if i not in placeholder_list]

check_separation(prey_in_front_etas_consumption, all_etas_consumption)
check_separation(prey_in_front_etas_scs, all_etas_scs)

prey_in_front = get_for_specific_neurons(ata, placeholder_list)
# predator_only = get_for_specific_neurons(ata, predator_only_ns)
# prey_only = get_for_specific_neurons(ata, prey_only_ns)
predator_cs = get_for_specific_neurons(ata, predator_cs_ns)
valence = get_for_specific_neurons(ata, valence_ns)
prey_full_field = get_for_specific_neurons(ata, prey_full_field_ns)

prey_in_front_std = [np.std([n/5000 for i, n in enumerate(ata[a]) if i in placeholder_list]) for a in ata.keys()]
prey_full_field_std = [np.std([n/5000 for i, n in enumerate(ata[a]) if i in prey_full_field_ns]) for a in ata.keys()]
valence_std = [np.std([n/5000 for i, n in enumerate(ata[a]) if i in valence_ns]) for a in ata.keys()]
all_std = [np.std([n/5000 for i, n in enumerate(ata[a])]) for a in ata.keys()]
used_actions = [0, 3, 4, 5]

stdss = []

for a in used_actions:
    stds = []
    stds.append(prey_in_front_std[a])
    stds.append(prey_full_field_std[a])
    stds.append(valence_std[a])
    stds.append(all_std[a])
    stdss.append(stds)

plot_average_action_scores_comparison([prey_in_front,prey_full_field, valence, ata, ], ["15o-in-Front", "Prey-Full-Field", "Valence", "All Neurons"], stdss)
#
# plot_average_action_scores_comparison([predator_only, prey_only, ata], ["Predator-Only","Prey-Only", "All"])
# Timeseries:

eta = get_full_eta_timeseries("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)
eta = np.absolute(eta)
prey_in_front_capture_average, std = get_average_timeseries(eta, placeholder_list)
prey_full_field, std2 = get_average_timeseries(eta, prey_full_field_ns)
predator_cs, std3 = get_average_timeseries(eta, predator_cs_ns)
other_capture_average, std23 = get_average_timeseries(eta, [i for i in range(512) if i not in placeholder_list and i not in prey_full_field_ns])
predator_only_capture_average, std33 = get_average_timeseries(eta, predator_only_ns)
display_multiple_average_eta_timeseries([prey_in_front_capture_average, other_capture_average, prey_full_field, predator_cs], ["15o-in-Front", "All Neurons", "Prey-Full-Field", "Predator-RW"], std, std2, std3)


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


