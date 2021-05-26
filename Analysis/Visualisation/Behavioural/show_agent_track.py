from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import math
import numpy as np
import random
import seaborn as sns


from Analysis.Behavioural.New.show_spatial_density import get_action_name
from Analysis.load_data import load_data
from Analysis.Behavioural.New.extract_event_action_sequence import get_escape_sequences, extract_predator_action_sequences, extract_consumption_action_sequences

def create_color_map(actions):
    colors = []
    for action in actions:
        if action == "RT Right" or action == "RT Left":
            colors.append("green")
        elif action == "sCS":
            colors.append("red")
        elif action == "J-turn Left" or action == "J-turn Right":
            colors.append("yellow")
        else:
            colors.append("purple")

    return colors


def colored_2d_track(position, action_choice, capture_positions=None):
    # Note that due to the inverse scale in the environment, should be rotated in y axis.
    data  = {}
    if capture_positions:
        data["x"] = [i[0] for i in position] + [i[0] for i in capture_positions]
        data["y"] = [i[1] for i in position] + [i[1] for i in capture_positions]
        data["Bout"] = [get_action_name(a) for a in action_choice] + ["Consumption" for i in capture_positions]
        colors = sns.color_palette("hls", len(set(action_choice)) + 1)
    else:
        data["x"] = [i[0] for i in position]
        data["y"] = [i[1] for i in position]
        data["Bout"] = [get_action_name(a) for a in action_choice]
        colors = sns.color_palette("hls", len(set(action_choice)))
    data["s"] = [10 for i in range(len(action_choice))]

    actions = []
    for action in data["Bout"]:
        if action not in actions:
            actions.append(action)
    colors = create_color_map(actions)

    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x="x", y="y", hue="Bout", data=data, palette=colors, s=100, alpha=1)
    plt.setp(ax.get_legend().get_title(), fontsize='25')
    plt.setp(ax.get_legend().get_texts(), fontsize='20')
    plt.xlim(0, 1500)
    plt.ylim(0, 1500)
    plt.show()


def colored_2d_track_turns(position, action_choice, orientation_changes):
    # Note that due to the inverse scale in the environment, should be rotated in y axis.
    data = {}
    # sns.set()
    turn_stamps = [i for i, a in enumerate(action_choice) if a == 1 or a == 2]
    data["x"] = [p[0] for i, p in enumerate(position) if i in turn_stamps]
    data["y"] = [p[1] for i, p in enumerate(position) if i in turn_stamps]
    data["Bout"] = [get_action_name(a) for i, a in enumerate(action_choice) if i in turn_stamps]
    data["s"] = [10 for i in range(len(action_choice))]
    data["Delta-Angle"] = [o for i, o in enumerate(orientation_changes) if i in turn_stamps]
    colors = ['b', 'g', 'g', 'r', 'y', 'y', "k", "m", "m"]

    # colors = sns.color_palette("hls", len(set(action_choice)))
    # colors = sns.color_palette("hls", 2)
    colors = ["green", "lightgreen"]
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x="x", y="y", hue="Bout", data=data, palette=colors, s=100, alpha=1)
    plt.setp(ax.get_legend().get_title(), fontsize='25')
    plt.setp(ax.get_legend().get_texts(), fontsize='20')
    plt.xlim(0, 1500)
    plt.ylim(0, 1500)
    plt.show()


# data = load_data("new_differential_prey_ref-6", "Behavioural-Data-Free-1", "Naturalistic-8")
# osition = data["position"]
# action_choice = data["behavioural choice"]

# for i in range(1, 10):
#     data = load_data("even_prey_ref-5", "Behavioural-Data-Free", f"Naturalistic-{i}")
#
#     position = data["position"]
#     action_choice = data["behavioural choice"]
#     ac, timestamps = extract_predator_action_sequences(data)
#     # timestamps = random.sample(timestamps, 20)
#     capture_timestamps = [i[-1] for i in timestamps]
#     capture_positions = [p for i, p in enumerate(position) if i in capture_timestamps]
#     #
#     ts = []
#     for i, t in enumerate(timestamps):
#         if 8 in ac[i] or 7 in ac[i]:
#             ts += t
#     position = [p for i, p in enumerate(position) if i in ts]
#     action_choice = [a for i, a in enumerate(action_choice) if i in ts]
#     colored_2d_track(position, action_choice)

# #
# data = load_data("even_prey_ref-3", "Behavioural-Data-Free", "Prey-1")
# # data = load_data("new_differential_prey_ref-6", "Behavioural-Data-Free-1", "Naturalistic-2")
# position = data["position"]
# action_choice = data["behavioural choice"]
# ac, timestamps = extract_consumption_action_sequences(data)
# timestamps = random.sample(timestamps, 10)
# capture_timestamps = [i[-1] for i in timestamps]
# capture_positions = [p for i, p in enumerate(position) if i in capture_timestamps]
# ts = []
# for t in timestamps:
#     ts += t
# position = [p for i, p in enumerate(position) if i in ts]
# action_choice = [a for i, a in enumerate(action_choice) if i in ts]
# action_choice = np.array(action_choice)
# position = np.array(position)
# #
# # # Shorten data
# # # position = position[200: 500]
# # # action_choice = action_choice[200: 500]
# #
# colored_2d_track(position, action_choice, capture_positions)
