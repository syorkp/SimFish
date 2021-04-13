import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from Analysis.load_data import load_data


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


def neuron_action_scores(event_triggered_averages):
    for n in range(len(event_triggered_averages["0"])):
        neuron_responses = [event_triggered_averages[str(i)][n] for i in range(10)]
        plt.bar([str(i) for i in range(10)], neuron_responses)
        plt.show()


def plot_average_action_scores(event_triggered_averages):
    mean_scores = []
    for a in range(10):
        m = np.mean(event_triggered_averages[str(a)])
        mean_scores.append(m)
    plt.bar([str(i) for i in range(10)], mean_scores)
    plt.show()


def get_for_specific_neurons(atas, neuron_list):
    subset = {}
    for key in atas.keys():
        subset[key] = [atas[key][i] for i in neuron_list]
    return subset


def shared_eta_distribution(event_triggered_averages_1, event_triggered_averages_2, labels):
    for a in range(10):
        atas = event_triggered_averages_1[str(a)]
        atas2 = event_triggered_averages_2[str(a)]
        if np.any(atas) or numpy.any(atas2):
            plt.figure(figsize=(5, 8))
            plt.hist([atas, atas2], color=["r", "b"])
            plt.title(f"ETAs for action {a}")
            plt.show()


def boxplot_of_etas(atas):
    labels = [key for key in atas.keys() if np.any(atas[key])]
    data = [atas[l] for l in labels]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=labels)
    ax.set_ylim([-1000, 1000])
    plt.show()


# data = load_data("even_prey_ref-5", "Naturalistic", "Naturalistic-1")
# ata = get_eta(data, "actions")
# boxplot_of_etas(ata)
# prey_only = [13, 21, 29, 44, 51, 52, 53, 57, 59, 60, 65, 69, 76, 86, 90, 115, 116, 121, 122, 129, 138, 142, 149, 157, 171, 173, 175, 176, 182, 183, 186, 188, 191, 201, 203, 205, 221, 225, 232, 239, 250, 260, 268, 292, 303, 312, 321, 327, 328, 347, 354, 362, 366, 368, 390, 395, 399, 402, 403, 406, 415, 429, 446, 447, 456, 463, 481, 497, 504]
# pred_only = [4, 5, 9, 10, 15, 16, 17, 22, 28, 34, 41, 42, 46, 47, 49, 55, 56, 64, 70, 77, 79, 85, 89, 93, 95, 98, 99, 100, 101, 106, 114, 118, 120, 133, 134, 135, 136, 139, 140, 145, 156, 158, 163, 166, 172, 174, 178, 179, 181, 189, 193, 194, 198, 200, 204, 206, 208, 213, 220, 224, 234, 236, 245, 249, 251, 253, 255, 257, 261, 263, 266, 269, 271, 275, 295, 296, 307, 317, 329, 338, 345, 346, 352, 364, 372, 381, 383, 385, 388, 418, 421, 424, 436, 449, 450, 453, 454, 458, 462, 468, 470, 482, 483, 489, 492, 495, 501, 503, 505, 509, 511]
# ata_subset_1 = get_for_specific_neurons(ata, prey_only)
# ata_subset_2 = get_for_specific_neurons(ata, pred_only)
# boxplot_of_etas(ata_subset_1)
# boxplot_of_etas(ata_subset_2)
# shared_eta_distribution(ata_subset_1, ata_subset_2, ["prey only", "pred only"])
# eta_distribution(ata_subset_1, "Prey-Only")
# eta_distribution(ata_subset_2, "Predator-Only")
# plot_all_action_scores(ata)
