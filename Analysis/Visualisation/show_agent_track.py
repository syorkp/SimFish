from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import math
from Analysis.load_data import load_data
import numpy as np
import seaborn as sns


def colored_2d_track(position, action_choice):
    # Note that due to the inverse scale in the environment, should be rotated in y axis.
    data  = {}
    data["x"] = [i[0] for i in position]
    data["y"] = [i[1] for i in position]
    data["a"] = [str(a) for a in action_choice]
    data["s"] = [10 for i in range(len(action_choice))]

    colors = sns.color_palette("hls", len(set(action_choice)))

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="x", y="y", hue="a", data=data, palette=colors, s=100, alpha=1)
    plt.xlim(0, 1500)
    plt.ylim(0, 1500)
    plt.show()


def colored_2d_track_turns(position, action_choice):
    # Note that due to the inverse scale in the environment, should be rotated in y axis.
    data = {}
    # sns.set()
    turn_stamps = [i for i, a in enumerate(action_choice) if a == 1 or a == 2]
    data["x"] = [p[0] for i, p in enumerate(position) if i in turn_stamps]
    data["y"] = [p[1] for i, p in enumerate(position) if i in turn_stamps]
    data["a"] = [str(a) for i, a in enumerate(action_choice) if i in turn_stamps]
    data["s"] = [10 for i in range(len(action_choice))]

    # colors = sns.color_palette("hls", len(set(action_choice)))
    colors = sns.color_palette("hls", 2)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="x", y="y", hue="a", data=data, palette=colors, s=100, alpha=1)
    # g = sns.legend(["RT Left", "RT Right"])
    # leg = g.axes.flat[0].get_legend()
    # for t, l in zip(leg.texts, ["RT Left", "RT Right"]): t.set_text(l)
    plt.legend([1, 2], ["RT Left", "RT Right"])
    plt.xlim(0, 1500)
    plt.ylim(0, 1500)
    plt.show()

# data = load_data("even_prey_ref-4", "Behavioural-Data-Free", "Prey-1")
# position = data["position"]
# action_choice = data["behavioural choice"]
#
# # Shorten data
# # position = position[200: 500]
# # action_choice = action_choice[200: 500]
#
# colored_2d_track(position, action_choice)
