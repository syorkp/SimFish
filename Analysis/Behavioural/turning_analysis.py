import matplotlib.pyplot as plt
import seaborn as sns

from Analysis.load_data import load_data
from Analysis.Visualisation.Behavioural.show_agent_track import colored_2d_track_turns
from scipy.interpolate import make_interp_spline, BSpline


def plot_turning_sequences(fish_angle):
    # sns.set()
    fish_angle = fish_angle.tolist()
    angle_changes = [fish_angle[i]-fish_angle[i-1] for i, angle in enumerate(fish_angle) if i!=0][-100:]
    # plt.bar(range(len(angle_changes)), angle_changes, color="blue")
    # plt.xlabel("Time (Step)")
    # plt.ylabel("Turn Amplitude (pi radians)")
    angles = {}
    angles["Time (Step)"] = [i for i in range(len(angle_changes))]
    angles["Turn Amplitude (pi radians)"] = angle_changes
    angles["Color"] = ["r" if angle < 0 else "b" for angle in angle_changes]
    ax = sns.barplot(x="Time (Step)", y="Turn Amplitude (pi radians)", hue="Color", data=angles)
    ax.get_legend().remove()
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.xlabel("Time (Step)")
    plt.ylabel("Turn Amplitude (pi radians)")
    plt.show()


def get_free_swimming_sequences(data):
    """Requires the following data: position, prey_positions, predator. Assumes square arena 1500."""
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    wall_timestamps = [i for i, p in enumerate(data["position"]) if 200 < p[0] < 1300 and 200<p[1]<1300]
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
    action_sequences = []
    current_action_sequence = []
    previous_point = 0
    for ts in desired_timestamps:
        if ts - 1 == previous_point:
            current_action_sequence.append(data["behavioural choice"][ts])
            previous_point = ts
        else:
            if previous_point == 0:
                current_action_sequence.append(data["behavioural choice"][ts])
                previous_point = ts
            else:
                action_sequences.append(current_action_sequence)
                current_action_sequence = [data["behavioural choice"][ts]]
                previous_point = ts
    if len(current_action_sequence) > 0:
        action_sequences.append(current_action_sequence)
    return action_sequences


def model_of_action_switching(sequences):
    switch_right_count = 0
    switch_left_count = 0
    total_left = 0
    total_right = 0
    left_durations = []
    right_durations = []
    for sequence in sequences:
        if len(sequence) < 5:
            continue
        if sequence[0] == 1:
            total_left += 1
        elif sequence[0] == 2:
            total_right += 1

        count = 0
        for i, a in enumerate(sequence[1:]):
            count += 1
            if a == 1:
                total_left += 1
                if sequence[i-1] != a:
                    left_durations.append(count)
                    count = 0
                    switch_right_count += 1
            elif a == 2:
                total_right += 1
                if sequence[i - 1] != a:
                    right_durations.append(count)
                    count = 0
                    switch_left_count += 1
    switch_right_p = switch_right_count/total_left
    switch_left_p = switch_left_count/total_right
    return switch_left_p, switch_right_p, left_durations, right_durations

from scipy.interpolate import UnivariateSpline
import numpy as np


def plot_switching_distribution(left_durs, right_durs):
    left_durs = [i for i in left_durs if i>1]
    right_durs = [i for i in right_durs if i>1]

    sns.set()
    seq_lengths = left_durs + right_durs
    p, x = np.histogram(seq_lengths, bins=10)  # bin it into n = N//10 bins
    x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    f = UnivariateSpline(x, p, s=10)
    plt.plot(x, f(x)/len(seq_lengths))
    plt.show()

    # p, x = np.histogram(left_durs, bins=10)  # bin it into n = N//10 bins
    # x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=10)
    # plt.plot(x, f(x)/len(left_durs))
    # plt.show()
    #
    # p, x = np.histogram(right_durs, bins=10)  # bin it into n = N//10 bins
    # x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=10)
    # plt.plot(x, f(x)/len(right_durs))
    # plt.show()


def new_switching_plot(action_sequences):
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
        average[i] = av/len(transformed_sequences)
    cum_average = [sum(average[:i]) for i, a in enumerate(average)]
    # spl = make_interp_spline(range(len(cum_average)), cum_average, k=2)  # type: BSpline
    # power_smooth = spl(np.linspace(0, 20, 10))
    plt.plot(cum_average)
    plt.show()


def divide_sequences(action_sequences):
    new_action_sequences = []
    for seq in action_sequences:
        while len(seq) > 10:
            new_action_sequences.append(seq[:10])
            seq = seq[10:]
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

# data = load_data("new_even_prey_ref-2", "Behavioural-Data-Free", f"Prey-1")
# colored_2d_track_turns(data["position"][300:425], data["behavioural choice"][300:425])
#
# for i in range(1, 10):
#     data = load_data("new_even_prey_ref-2", "Behavioural-Data-Free", f"Prey-{i}")
#     colored_2d_track_turns(data["position"][100:500], data["behavioural choice"][100:500])

orientation_log = []
action_sequences = []
for j in range(1, 4):
    for i in range(1, 11):
        data = load_data("new_differential_prey_ref-4", f"Behavioural-Data-Free-{j}", f"Naturalistic-{i}")
        new_as = get_free_swimming_sequences(data)
        action_sequences += [[a for a in seq if a == 1 or a == 2] for seq in new_as]
        orientation_changes = [data["fish_angle"][i]-data["fish_angle"][i-1] for i, angle in enumerate(data["fish_angle"]) if i!=0]
        orientation_log = orientation_log + orientation_changes
        # colored_2d_track_turns(data["position"][-200:], data["behavioural choice"][-200:], orientation_changes[-200:])
        # plot_turning_sequences(data["fish_angle"])

# for j in range(1, 4):
#     for i in range(1, 11):
#         data = load_data("new_differential_prey_ref-3", f"Behavioural-Data-Free-{j}", f"Naturalistic-{i}")
#         new_as = get_free_swimming_sequences(data)
#         action_sequences += [[a for a in seq if a == 1 or a == 2] for seq in new_as]
#         orientation_changes = [data["fish_angle"][i]-data["fish_angle"][i-1] for i, angle in enumerate(data["fish_angle"]) if i!=0]
#         orientation_log = orientation_log + orientation_changes

action_sequences = get_frameshift_sequences(action_sequences)
action_sequences = divide_sequences(action_sequences)
new_switching_plot(action_sequences)
l, r, sl, sr = model_of_action_switching(action_sequences)
plot_switching_distribution(sl, sr)

# plot_turning_sequences(data["fish_angle"])
# colored_2d_track_turns(data["position"][-200:], data["behavioural choice"][-200:])
x = True
