import numpy
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
    for ata in atas:
        mean_scores = []
        for a in range(10):
            m = np.mean(ata[str(a)])
            mean_scores.append(m)
        atas2.append(mean_scores)

    plt.figure(figsize=(15, 5))
    df = pd.DataFrame({label: data for data, label in zip(atas2, labels)})

    df.plot.bar(rot=0, figsize=(15, 5))  # , color={labels[0]: "blue", labels[1]: "blue", labels[2]: "red"}
    plt.xlabel("Action-Triggered Average")
    plt.ylabel("Action")
    plt.xticks([a for a in range(10)], [get_action_name(a) for a in range(10)])
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



# boxplot_of_etas(ata)
# 4
prey_only_4 = [33, 52, 65, 247, 311, 376, 486]
pred_only_4 = [4, 7, 12, 13, 15, 19, 20, 23, 24, 25, 26, 27, 28, 43, 45, 46, 47, 48, 50, 51, 53, 55, 56, 61, 62, 63, 66, 67, 69, 75, 76, 77, 78, 80, 82, 83, 85, 88, 89, 90, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 106, 107, 108, 111, 118, 119, 120, 121, 124, 125, 126, 127, 128, 130, 131, 135, 137, 139, 144, 147, 153, 156, 157, 159, 163, 167, 168, 173, 184, 185, 191, 192, 193, 197, 201, 203, 205, 206, 209, 211, 213, 214, 215, 219, 220, 221, 223, 224, 225, 227, 228, 229, 231, 232, 235, 238, 244, 245, 246, 251, 254, 255, 258, 263, 267, 280, 282, 286, 288, 289, 293, 299, 306, 312, 315, 318, 320, 321, 322, 337, 351, 353, 354, 356, 362, 364, 377, 394, 398, 399, 400, 407, 409, 410, 412, 419, 426, 437, 447, 465, 470, 472, 481, 484, 489, 492, 503]
prey_in_front_4 = [3, 9, 10, 30, 35, 38, 42, 52, 58, 59, 68, 87, 103, 109, 113, 117, 134, 142, 145, 148, 149, 162, 164, 171, 172, 175, 176, 179, 180, 181, 189, 190, 198, 207, 216, 222, 226, 236, 239, 241, 243, 253, 256, 257, 276, 283, 287, 292, 297, 305, 308, 309, 313, 316, 333, 334, 335, 344, 346, 357, 376, 381, 383, 384, 386, 388, 391, 396, 397, 401, 403, 421, 422, 429, 430, 436, 439, 443, 445, 446, 455, 461, 473, 490, 495, 497, 498, 508, 510, 511]

# 5
prey_only_5 = [8, 13, 29, 44, 51, 52, 53, 59, 60, 65, 69, 76, 86, 90, 98, 116, 121, 122, 129, 134, 138, 142, 149, 157, 171, 173, 176, 182, 184, 188, 191, 201, 205, 217, 221, 232, 239, 250, 260, 289, 315, 328, 329, 332, 347, 362, 385, 390, 395, 399, 402, 406, 411, 415, 419, 429, 436, 446, 469, 481, 488, 497, 504]
pred_only_5 = [5, 9, 15, 16, 17, 22, 41, 42, 46, 47, 56, 57, 70, 73, 77, 85, 89, 93, 95, 99, 100, 101, 106, 118, 120, 123, 127, 133, 136, 139, 145, 158, 163, 166, 172, 174, 178, 179, 181, 183, 190, 193, 194, 199, 200, 204, 208, 213, 220, 234, 236, 249, 251, 253, 255, 261, 263, 266, 269, 271, 277, 295, 296, 303, 307, 321, 338, 342, 364, 372, 381, 382, 383, 409, 424, 449, 450, 454, 456, 461, 462, 470, 474, 483, 495, 496, 501, 503, 505, 509, 511]
prey_in_front_5 = [0, 6, 11, 21, 26, 27, 29, 31, 44, 48, 63, 71, 82, 87, 102, 105, 107, 109, 113, 115, 117, 121, 128, 137, 140, 141, 143, 144, 147, 153, 155, 159, 160, 165, 180, 182, 185, 197, 203, 205, 207, 209, 210, 215, 218, 227, 228, 240, 242, 243, 246, 247, 254, 268, 270, 273, 274, 278, 280, 283, 286, 287, 297, 298, 306, 308, 310, 311, 313, 322, 323, 324, 325, 332, 333, 334, 335, 341, 343, 344, 350, 351, 352, 354, 355, 361, 362, 366, 368, 374, 376, 397, 400, 402, 403, 412, 413, 416, 419, 423, 425, 426, 427, 428, 429, 430, 433, 435, 437, 439, 440, 441, 442, 446, 447, 448, 455, 459, 464, 466, 472, 473, 475, 476, 477, 482, 485, 486, 491, 493, 499, 500, 510]

# 6
prey_only_6 = [284, 309, 343, 403, 475, 496, 511]
pred_only_6 = [2, 8, 9, 10, 18, 21, 27, 28, 37, 40, 47, 63, 64, 72, 86, 89, 93, 104, 105, 107, 112, 119, 129, 132, 144, 147, 150, 159, 162, 167, 174, 187, 192, 193, 201, 205, 206, 211, 212, 213, 224, 227, 244, 245, 256, 272, 273, 292, 297, 304, 311, 315, 326, 337, 345, 354, 389, 390, 397, 404, 419, 427, 428, 436, 437, 440, 445, 448, 450, 453, 461, 465, 473, 479, 486, 487, 495, 503, 508]
prey_in_front_6 =[262, 7, 392, 14, 17, 20, 281, 282, 293, 171, 48, 178, 179, 308, 309, 56, 185, 186, 60, 319, 322, 451, 69, 197, 76, 207, 80, 336, 338, 83, 470, 90, 353, 98, 483, 101, 357, 485, 370, 373, 374, 247, 123, 511]

# 7
prey_only_7 = [3, 135, 150, 173, 178, 203, 206, 232, 234, 253, 279, 281, 312, 319, 341, 349, 379, 393, 403, 439, 454, 464, 498]
pred_only_7 = [1, 23, 25, 31, 37, 38, 39, 44, 45, 50, 57, 60, 67, 70, 71, 73, 78, 80, 81, 84, 86, 89, 90, 91, 101, 102, 104, 107, 114, 124, 144, 146, 152, 153, 163, 166, 179, 180, 186, 187, 188, 195, 198, 199, 200, 212, 220, 224, 229, 233, 240, 249, 255, 271, 277, 280, 287, 293, 304, 308, 309, 318, 353, 354, 359, 365, 367, 368, 371, 372, 377, 380, 384, 389, 398, 416, 423, 426, 438, 443, 445, 463, 465, 473, 478, 483, 492, 493, 494, 499, 501, 505, 511]
prey_in_front_7 =[4, 5, 14, 15, 17, 21, 22, 30, 35, 48, 55, 58, 63, 64, 66, 74, 75, 82, 92, 95, 97, 99, 111, 113, 116, 118, 123, 136, 137, 140, 142, 147, 148, 149, 156, 158, 167, 170, 182, 184, 194, 197, 201, 205, 207, 209, 215, 225, 226, 235, 238, 248, 252, 266, 279, 292, 294, 301, 305, 311, 316, 323, 328, 331, 333, 336, 337, 338, 339, 346, 351, 363, 366, 369, 375, 388, 390, 397, 404, 405, 410, 411, 418, 420, 425, 428, 432, 435, 436, 440, 442, 444, 450, 451, 458, 466, 469, 470, 479, 480, 484, 485, 488, 500, 502, 510]


# data = load_data("even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic-7")
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
