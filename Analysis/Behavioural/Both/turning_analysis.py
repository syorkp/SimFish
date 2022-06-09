import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.extract_exploration_sequences import extract_exploration_action_sequences_with_positions
from Analysis.Behavioural.Tools.extract_turn_sequences import extract_turn_sequences, extract_purely_turn_sequences
from Analysis.Behavioural.Tools.extract_exploration_sequences import extract_exploration_action_sequences_with_fish_angles
from Analysis.Behavioural.Tools.extract_exploration_sequences import get_no_prey_stimuli_sequences, get_exploration_sequences


def plot_turning_sequences(fish_angle):
    # sns.set()

    angle_changes = [fish_angle[i]-fish_angle[i-1] for i, angle in enumerate(fish_angle) if i!=0][-100:]
    # plt.bar(range(len(angle_changes)), angle_changes, color="blue")
    # plt.xlabel("Time (Step)")
    # plt.ylabel("Turn Amplitude (pi radians)")
    angles = {}
    angles["Time (Step)"] = [i for i in range(len(angle_changes))]
    angles["Turn Amplitude (pi radians)"] = angle_changes
    angles["Color"] = ["r" if angle < 0 else "b" for angle in angle_changes]
    fig = plt.figure()
    ax = sns.barplot(x="Time (Step)", y="Turn Amplitude (pi radians)", hue="Color", data=angles)
    ax.get_legend().remove()
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.xlabel("Time (Step)", fontsize=18)
    plt.ylabel("Turn Amplitude (pi radians)", fontsize=18)
    fig.tight_layout()
    plt.show()


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


def compute_cumulative_probability(sequence_lengths):
    max_duration = max(sequence_lengths)
    count_seq_length = np.zeros((max_duration+1))
    for sequence in sequence_lengths:
        count_seq_length[sequence] += 1
    cumulative_count_sequence_length = [sum(count_seq_length[:i]) for i in range(1, len(count_seq_length+1))]
    cumulative_probability = np.array(cumulative_count_sequence_length)/max(cumulative_count_sequence_length)
    return cumulative_probability


def cumulative_switching_probability_plot(left_durs, right_durs, left_durs2, right_durs2, label):
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

    plt.plot(x, cdf, label="Agent")
    plt.plot(x2, cdf2, label="Random Switching")
    plt.xlabel("Turn Streak Length", fontsize=18)
    plt.ylabel("Cumulative Probability", fontsize=18)
    plt.title(label)
    plt.legend()
    plt.show()


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
    plt.fill_between(range(11), ermin, ermax, color="b", alpha=0.5)
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
    plt.plot(cum_average)
    plt.xlabel("Number of Turns", fontsize=20)
    plt.ylabel("Cumulative Turn Direction", fontsize=20)
    plt.hlines(0, 0, 10, color="r")
    plt.title(label)
    # plt.fill_between(range(11), err_min, err_max)
    plt.show()


def divide_sequences(action_sequences):
    new_action_sequences = []
    for seq in action_sequences:
        while len(seq) > 11:
            new_action_sequences.append(seq[:11])
            seq = seq[11:]
        new_action_sequences.append(seq)
    return new_action_sequences


def get_frameshift_sequences(action_sequences):
    new_sequences = []
    for seq in action_sequences:
        new_sequences.append(seq)
        for i in range(1, len(seq)-8):
            if seq[i] != seq[-1]:
                new_sequences.append(seq[i:])
    return new_sequences


def randomly_switching_fish(n_sequences=100):
    sequences = []
    for i in range(n_sequences):
        seq = np.random.randint(2, size=10)
        seq = [1 if i == 1 else 2 for i in seq]
        sequences.append(seq)
    return model_of_action_switching(sequences)


def plot_all_turn_analysis(model_name, assay_config, assay_id, n, use_purely_turn_sequences=True):
    no_prey_actions, no_prey_timestamps = get_no_prey_stimuli_sequences(model_name, assay_config, assay_id, n)
    otherwise_exploration_sequences = get_exploration_sequences(model_name, assay_config, assay_id, n)

    if use_purely_turn_sequences:
        turn_exploration_sequences = extract_purely_turn_sequences(otherwise_exploration_sequences, 5)
        turn_no_prey_sequences = extract_purely_turn_sequences(no_prey_actions, 5)
    else:
        turn_exploration_sequences = extract_turn_sequences(otherwise_exploration_sequences)
        turn_no_prey_sequences = extract_turn_sequences(no_prey_actions)

    # Cumulative turn direction plots:
    cumulative_turn_direction_plot(turn_no_prey_sequences,
                                   label=f"Cumulative Turn Direction (no prey near, only turns) {model_name}")
    cumulative_turn_direction_plot(turn_exploration_sequences,
                                   label=f"Cumulative Turn Direction (no prey or walls, only turns) {model_name}")

    # Cumulative probability plot.
    l, r, sl, sr = model_of_action_switching(turn_exploration_sequences)
    l2, r2, sl2, sr2 = randomly_switching_fish()
    cumulative_switching_probability_plot(sl, sr, sl2, sr2, label=f"Cumulative Switching Probability (exploration) {model_name}")

    l, r, sl, sr = model_of_action_switching(turn_no_prey_sequences)
    l2, r2, sl2, sr2 = randomly_switching_fish()
    cumulative_switching_probability_plot(sl, sr, sl2, sr2, label=f"Cumulative Switching Probability (no prey) {model_name}")



# sl_compiled = []
# sr_compiled = []
# sl2_compiled = []
# sr2_compiled = []
# turn_exploration_sequences_compiled = []
# for i in range(1, 5):
#     data = load_data(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", f"Naturalistic-1")
#
#     # Cumulative turn direction plot
#     exploration_timestamps, exploration_sequences, exploration_fish_positions = extract_exploration_action_sequences_with_positions(data)
#     turn_exploration_sequences = extract_turn_sequences(exploration_sequences)
#     cumulative_turn_direction_plot(turn_exploration_sequences)
#
#     # Orientation plot
#     exploration_timestamps, exploration_sequences, exploration_fish_orientations = extract_exploration_action_sequences_with_fish_angles(data)
#     plot_turning_sequences(exploration_fish_orientations[-2])
#
#     # Cumulative probability plot.
#     l, r, sl, sr = model_of_action_switching(turn_exploration_sequences)
#     l2, r2, sl2, sr2 = randomly_switching_fish()
#     cumulative_switching_probability_plot(sl, sr, sl2, sr2)
#
#     turn_exploration_sequences_compiled += turn_exploration_sequences
#     sl_compiled += sl
#     sr_compiled += sr
#     sl2_compiled += sl2
#     sr2_compiled += sr2
#
# cumulative_switching_probability_plot(sl_compiled, sr_compiled, sl2_compiled, sr2_compiled)
# cumulative_turn_direction_plot(turn_exploration_sequences_compiled)


sl_compiled = []
sr_compiled = []
sl2_compiled = []
sr2_compiled = []
turn_exploration_sequences_compiled = []

# Exploration sequences based on visual stimulation level
for i in range(1, 5):
    plot_all_turn_analysis(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", f"Naturalistic", 10)

    # turn_exploration_sequences_compiled += turn_exploration_sequences
    # sl_compiled += sl
    # sr_compiled += sr
    # sl2_compiled += sl2
    # sr2_compiled += sr2

# cumulative_switching_probability_plot(sl_compiled, sr_compiled, sl2_compiled, sr2_compiled, f"Cumulative Switching Probability Plot all models")
# cumulative_turn_direction_plot(turn_exploration_sequences_compiled, f"Cumulative turn direction plot all models")

# VERSION 1

# for i in range(1, 10):
#     data = load_data("new_even_prey_ref-2", "Behavioural-Data-Free", f"Prey-{i}")
#     colored_2d_track_turns(data["position"][100:500], data["behavioural choice"][100:500])
#
# orientation_log = []
# action_sequences = []
# for j in range(1, 4):
#     for i in range(1, 11):
#         data = load_data("new_differential_prey_ref-3", f"Behavioural-Data-Free-{j}", f"Naturalistic-{i}")
#         new_as = get_free_swimming_sequences(data)
#         action_sequences += [[a for a in seq if a == 1 or a == 2] for seq in new_as]
#         orientation_changes = [data["fish_angle"][i]-data["fish_angle"][i-1] for i, angle in enumerate(data["fish_angle"]) if i!=0]
#         orientation_log = orientation_log + orientation_changes
#         # colored_2d_track_turns(data["position"][-200:], data["behavioural choice"][-200:], orientation_changes[-200:])
#         plot_turning_sequences(data["fish_angle"])

# all_action_sequences = []
# for x in [3, 4]:
#     action_sequences = []
#     for j in range(1, 4):
#         for i in range(1, 11):
#             data = load_data(f"new_differential_prey_ref-{x}", f"Behavioural-Data-Free-{j}", f"Naturalistic-{i}")
#             new_as = get_free_swimming_sequences(data)
#             action_sequences += [[a for a in seq if a == 1 or a == 2] for seq in new_as]
#             orientation_changes = [data["fish_angle"][i]-data["fish_angle"][i-1] for i, angle in enumerate(data["fish_angle"]) if i!=0]
#             orientation_log = orientation_log + orientation_changes
#     action_sequences = divide_sequences(action_sequences)
#     all_action_sequences.append(action_sequences)
#     l, r, sl, sr = model_of_action_switching(action_sequences)
#     l2, r2, sl2, sr2 = randomly_switching_fish()
#     plot_switching_distribution(sl, sr, sl2, sr2)
# # action_sequences = get_frameshift_sequences(action_sequences)
#
# new_switching_plot2(all_action_sequences)


# plot_turning_sequences(data["fish_angle"])
# colored_2d_track_turns(data["position"][-200:], data["behavioural choice"][-200:])
