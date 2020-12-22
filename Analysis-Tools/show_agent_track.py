import json

from matplotlib import pyplot as plt
import matplotlib.cm as cm

from load_data import load_data


# TODO: Make sure data load method fits.
data = load_data("Prey Stimuli", "Visual-Stimulus-Assay-2")


position = data["position"]
action_choice = data["behavioural choice"]

# TODO: Ensure is the right way up in future.

# Shorten data
position = position[200: 500]
action_choice = action_choice[200: 500]


def plot_2d_track():
    # Note that due to the inverse scale in the environment, should be rotated in y axis.
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_zlabel('Time')

    x = [i[0] for i in position]
    y = [i[1] for i in position]

    ax.plot(x, y)
    plt.xlim(0, 1000)
    plt.ylim(0, 700)
    # plt.gca().invert_yaxis()
    plt.show()


def colored_2d_track():
    x = [i[0] for i in position]
    y = [i[1] for i in position]

    cmap = cm.jet

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    c = action_choice
    ax.scatter(x, y, c=c, cmap=cmap)
    ax.set_xlabel("X Position (arbitrary distance units)", fontsize=12)
    ax.set_ylabel("Y Position (arbitrary distance units)", fontsize=12)
    ax.tick_params(labelsize=12)

    plt.xlim(0, 1000)
    plt.ylim(0, 700)
    plt.show()


# plot_2d_track()

colored_2d_track()
