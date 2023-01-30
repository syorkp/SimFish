import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import extract_exploration_action_sequences_with_fish_angles


def cumulative_turn_direction_plot_multiple_models(action_sequences):
    cum_averages = []

    for action_sequence in action_sequences:
        action_sequence = [seq for seq in action_sequence if len(seq) > 8]
        transformed_sequences = []
        mxln = 0

        for sequence in action_sequence:
            trans = [1 if a == sequence[0] else -1 for a in sequence]
            if len(trans) > mxln:
                mxln = len(trans)
            trans.pop(0)
            transformed_sequences.append(trans)
        average = [0 for i in range(mxln)]
        for sequence in transformed_sequences:
            for i, action in enumerate(sequence):
                average[i] += action
        for i, av in enumerate(average):
            average[i] = av / (len(transformed_sequences) * 2)
        cum_average = [sum(average[:i]) for i, a in enumerate(average)]
        cum_averages.append(cum_average)

    ermin = [min([cum_averages[m][i] for m, model in enumerate(cum_averages)]) for i, s in enumerate(cum_averages[0])]
    ermax = [max([cum_averages[m][i] for m, model in enumerate(cum_averages)]) for i, s in enumerate(cum_averages[0])]
    mean = [np.mean([cum_averages[m][i] for m, model in enumerate(cum_averages)]) for i, s in enumerate(cum_averages[0])]

    fig = plt.figure()
    sns.set()
    plt.plot(mean, color="orange")
    plt.xlabel("Number of Turns", fontsize=18)
    plt.ylabel("Cumulative Turn Direction", fontsize=18)
    plt.hlines(0, 0, 10, color="r")
    plt.fill_between(range(len(ermin)), ermin, ermax, color="b", alpha=0.5)
    fig.tight_layout()
    plt.show()


def cumulative_turn_direction_plot(action_sequences, label):
    action_sequences = [seq for seq in action_sequences if len(seq) > 8]
    transformed_sequences = []
    mxln = 0

    for sequence in action_sequences:
        trans = [1 if a == sequence[0] else -1 for a in sequence]
        if len(trans) > mxln:
            mxln = len(trans)
        trans.pop(0)
        transformed_sequences.append(trans)
    average = [0 for i in range(mxln)]
    for sequence in transformed_sequences:
        for i, action in enumerate(sequence):
            average[i] += action
    for i, av in enumerate(average):
        average[i] = av/(len(transformed_sequences)*2)
    cum_average = [sum(average[:i]) for i, a in enumerate(average)]
    # spl = make_interp_spline(range(len(cum_average)), cum_average, k=2)  # type: BSpline
    # power_smooth = spl(np.linspace(0, 20, 10))
    sns.set()
    plt.figure(figsize=(10, 10))
    plt.plot(cum_average)
    plt.xlabel("Number of Turns", fontsize=20)
    plt.ylabel("Cumulative Turn Direction", fontsize=20)
    plt.hlines(0, 0, 10, color="r")
    plt.title(label)
    try:
        plt.savefig(f"../../../Analysis-Output/Behavioural/Turn-Chains/{label}")
    except FileNotFoundError:
        try:
            plt.savefig(f"../../Analysis-Output/Behavioural/Turn-Chains/{label}")
        except FileNotFoundError:
            try:
                plt.savefig(f"../Analysis-Output/Behavioural/Turn-Chains/{label}")
            except FileNotFoundError:
                plt.savefig(f"./Analysis-Output/Behavioural/Turn-Chains/{label}")

    plt.clf()
    plt.close()
    # plt.fill_between(range(11), err_min, err_max)
    # plt.clf()


def cumulative_switching_probability_plot_multiple_models(left_durs_list, right_durs_list, left_durs2, right_durs2,
                                                          label, save_figure=True):
    """Given two sets of switching latencies, one random and one from a model, plots the cumulative probability of
    switching direction."""
    # left_durs = [i for i in left_durs if i>1]
    # right_durs = [i for i in right_durs if i>1]
    # left_durs2 = [i for i in left_durs2 if i>1]
    # right_durs2 = [i for i in right_durs2 if i>1]

    seq_lengths2 = left_durs2 + right_durs2
    cdf2 = compute_cumulative_probability(seq_lengths2)
    x2 = range(0, max(seq_lengths2))

    cdfs = []
    xs = []
    min_len = 1000
    for left_durs, right_durs in zip(left_durs_list, right_durs_list):
        if len(left_durs) == 0 or len(right_durs) == 0:
            return

        sns.set()
        seq_lengths = left_durs + right_durs

        cdf = compute_cumulative_probability(seq_lengths)
        cdfs.append(cdf)

        x = range(0, max(seq_lengths))
        xs.append(x)
        if max(seq_lengths) < min_len:
            min_len = max(seq_lengths)

    ermin = [min([cdfs[m][i] for m, model in enumerate(cdfs)]) for i in range(min_len)]
    ermax = [max([cdfs[m][i] for m, model in enumerate(cdfs)]) for i in range(min_len)]

    # plt.plot(x, cdf, label="Agent")
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.fill_between(range(len(ermin)), ermin, ermax, label="Agents")
    ax.plot(x2, cdf2, label="Random Switching", color="orange", linewidth=5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    plt.xlabel("Turn Streak Length", fontsize=30)
    plt.ylabel("Cumulative Probability", fontsize=30)
    plt.title(label)
    ax.legend(prop={'size': 30}, loc="lower right")
    if save_figure:
        try:
            plt.savefig(f"../../../Analysis-Output/Behavioural/Turn-Chains/{label}")
        except FileNotFoundError:
            try:
                plt.savefig(f"../../Analysis-Output/Behavioural/Turn-Chains/{label}")
            except FileNotFoundError:
                try:
                    plt.savefig(f"../Analysis-Output/Behavioural/Turn-Chains/{label}")
                except FileNotFoundError:
                    plt.savefig(f"./Analysis-Output/Behavioural/Turn-Chains/{label}")
    plt.clf()



def model_of_action_switching(sequences):
    switch_right_count = 0
    switch_left_count = 0
    total_left = 0
    total_right = 0
    left_durations = []
    right_durations = []
    for sequence in sequences:
        if sequence[0] == 1 or sequence[0] == 4:
            total_left += 1
        elif sequence[0] == 2 or sequence[0] == 5:
            total_right += 1

        count = 0
        for i, a in enumerate(sequence[1:]):
            count += 1
            if a == 1 or a == 4:
                total_left += 1
                if sequence[i-1] != a:
                    left_durations.append(count)
                    count = 0
                    switch_right_count += 1
            elif a == 2 or a == 5:
                total_right += 1
                if sequence[i - 1] != a:
                    right_durations.append(count)
                    count = 0
                    switch_left_count += 1
        if count > 0:
            if a == 1 or a == 4:
                left_durations.append(count)
            elif a == 2 or a == 5:
                right_durations.append(count)

    if total_left > 0 and total_right > 0:
        switch_right_p = switch_right_count/total_left
        switch_left_p = switch_left_count/total_right
        return switch_left_p, switch_right_p, left_durations, right_durations
    else:
        return 0, 0, [], []


def plot_turning_sequences(fish_angle, save_figure=True):
    # sns.set()

    angle_changes = [fish_angle[i]-fish_angle[i-1] for i, angle in enumerate(fish_angle) if i!=0][-100:]
    # plt.bar(range(len(angle_changes)), angle_changes, color="blue")
    # plt.xlabel("Time (Step)")
    # plt.ylabel("Turn Amplitude (radians)")
    angles = {}
    angles["Time (Step)"] = [i for i in range(len(angle_changes))]
    angles["Turn Amplitude (radians)"] = angle_changes
    angles["Color"] = ["r" if angle < 0 else "b" for angle in angle_changes]
    fig = plt.figure()
    ax = sns.barplot(x="Time (Step)", y="Turn Amplitude (radians)", hue="Color", data=angles)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.get_legend().remove()
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.xlabel("Time (Step)", fontsize=18)
    plt.ylabel("Turn Amplitude (radians)", fontsize=18)
    fig.tight_layout()
    if save_figure:
        try:
            plt.savefig(f"../../../Analysis-Output/Behavioural/Turn-Chains/orientation_plot.jpg")
        except FileNotFoundError:
            try:
                plt.savefig(f"../../Analysis-Output/Behavioural/Turn-Chains/orientation_plot.jpg")
            except FileNotFoundError:
                try:
                    plt.savefig(f"../Analysis-Output/Behavioural/Turn-Chains/orientation_plot.jpg")
                except FileNotFoundError:
                    plt.savefig(f"./Analysis-Output/Behavioural/Turn-Chains/orientation_plot.jpg")
    plt.show()


def compute_cumulative_probability(sequence_lengths):
    max_duration = max(sequence_lengths)
    count_seq_length = np.zeros((max_duration+1))
    for sequence in sequence_lengths:
        count_seq_length[sequence] += 1
    cumulative_count_sequence_length = [sum(count_seq_length[:i]) for i in range(1, len(count_seq_length+1))]
    cumulative_probability = np.array(cumulative_count_sequence_length)/max(cumulative_count_sequence_length)
    return cumulative_probability


def randomly_switching_fish(n_sequences=100):
    sequences = []
    for i in range(n_sequences):
        seq = np.random.randint(2, size=10)
        seq = [1 if i == 1 else 2 for i in seq]
        sequences.append(seq)
    return model_of_action_switching(sequences)


def randomly_switching_fish_new(real_fish_actions):
    sequence_lengths = [len(seq) for seq in real_fish_actions]
    flattened_real_fish_actions = [item for seq in real_fish_actions for item in seq]
    random.shuffle(flattened_real_fish_actions)
    random_sequences = []
    i = 0
    for l in sequence_lengths:
        i2 = i
        i += l
        random_sequences.append(flattened_real_fish_actions[i2: i])
    return model_of_action_switching(random_sequences)


def cumulative_switching_probability_plot(left_durs, right_durs, left_durs2, right_durs2, save_location):
    """Given two sets of switching latencies, one random and one from a model, plots the cumulative probability of
    switching direction."""
    # left_durs = [i for i in left_durs if i>1]
    # right_durs = [i for i in right_durs if i>1]
    # left_durs2 = [i for i in left_durs2 if i>1]
    # right_durs2 = [i for i in right_durs2 if i>1]
    if len(left_durs) == 0 or len(right_durs) == 0:
        return

    sns.set()
    seq_lengths = left_durs + right_durs
    seq_lengths2 = left_durs2 + right_durs2

    cdf = compute_cumulative_probability(seq_lengths)
    cdf2 = compute_cumulative_probability(seq_lengths2)
    x = range(0, max(seq_lengths))
    x2 = range(0, max(seq_lengths2))

    plt.figure(figsize=(10, 10))
    plt.plot(x, cdf, label="Agent")
    plt.plot(x2, cdf2, label="Random Switching")
    plt.xlabel("Turn Streak Length", fontsize=30)
    plt.ylabel("Cumulative Probability", fontsize=30)
    plt.legend(["Models", "Random Switching"], fontsize=30)
    try:
        plt.savefig(f"../../../Analysis-Output/Behavioural/Turn-Chains/{save_location}")
    except FileNotFoundError:
        try:
            plt.savefig(f"../../Analysis-Output/Behavioural/Turn-Chains/{save_location}")
        except FileNotFoundError:
            try:
                plt.savefig(f"../Analysis-Output/Behavioural/Turn-Chains/{save_location}")
            except FileNotFoundError:
                plt.savefig(f"./Analysis-Output/Behavioural/Turn-Chains/{save_location}")
    plt.clf()
    plt.close()
