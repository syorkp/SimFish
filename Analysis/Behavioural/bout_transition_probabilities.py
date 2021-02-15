import numpy as np
import matplotlib.pyplot as plt

from MarkovChain.markovchain import MarkovChain
import transitionMatrix


import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import numpy as np
import pandas as pd

import transitionMatrix as tm
from transitionMatrix import source_path
from transitionMatrix.estimators import cohort_estimator as es


from Analysis.load_data import load_data


def get_first_order_transition_counts(p1, p2, p3, n):
    transition_counts = np.zeros((10, 10))
    for file_index in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{file_index}")
        for i, a in enumerate(data["behavioural choice"]):
            if i == 0:
                pass
            else:
                transition_counts[data["behavioural choice"][i-1]][a] += 1

    return transition_counts


def get_second_order_transition_counts(p1, p2, p3, n):
    transition_counts = np.zeros((10, 10, 10))
    for file_index in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{file_index}")
        for i, a in enumerate(data["behavioural choice"]):
            if i == 0 or i == 1:
                pass
            else:
                transition_counts[data["behavioural choice"][i-2]][data["behavioural choice"][i-1]][a] += 1

    return transition_counts

def get_third_order_transition_counts(p1, p2, p3, n):
    transition_counts = np.zeros((10, 10, 10, 10))
    for file_index in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{file_index}")
        for i, a in enumerate(data["behavioural choice"]):
            if i == 0 or i == 1 or i == 2:
                pass
            else:
                transition_counts[data["behavioural choice"][i-3]][data["behavioural choice"][i-2]][data["behavioural choice"][i-1]][a] += 1

    return transition_counts

def get_transition_probabilities(transition_counts):
    transition_probabilities = transition_counts/np.sum(transition_counts)
    return transition_probabilities


def visualise_first_order_transitions(transition_probabilities):
    matrix_tran = transition_probabilities[:4, :4]
    mc = MarkovChain(matrix_tran, [str(i) for i in range(0, 4)])
    mc.draw()


def visualisation_method_2(transition_probabilities):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    plt.style.use(['ggplot'])
    plt.ylabel('From State')
    plt.xlabel('To State')
    mymap = plt.get_cmap("RdYlGn")
    mymap = plt.get_cmap("Reds")
    # mymap = plt.get_cmap("Greys")
    normalize = mpl.colors.LogNorm(vmin=0.0001, vmax=1)

    matrix_size = transition_probabilities.shape[0]
    square_size = 1.0 / matrix_size

    diagonal = transition_probabilities.diagonal()
    # colors = []

    ax.set_xticklabels(range(0, matrix_size))
    ax.set_yticklabels(range(0, matrix_size))
    ax.xaxis.set_ticks(np.arange(0 + 0.5 * square_size, 1 + 0.5 * square_size, square_size))
    ax.yaxis.set_ticks(np.arange(0 + 0.5 * square_size, 1 + 0.5 * square_size, square_size))

    # iterate over all elements of the matrix

    for i in range(0, matrix_size):
        for j in range(0, matrix_size):
            if transition_probabilities[i, j] > 0:
                rect_size = np.sqrt(transition_probabilities[i, j]) * square_size
            else:
                rect_size = 0

            dx = 0.5 * (square_size - rect_size)
            dy = 0.5 * (square_size - rect_size)
            p = patches.Rectangle(
                (i * square_size + dx, j * square_size + dy),
                rect_size,
                rect_size,
                fill=True,
                color=mymap(normalize(transition_probabilities[i, j]))
            )
            ax.add_patch(p)

    cbax = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    cb = mpl.colorbar.ColorbarBase(cbax, cmap=mymap, norm=normalize, orientation='vertical')
    cb.set_label("Transition Probability", rotation=270, labelpad=15)

    plt.show(block=True)
    plt.interactive(False)

#
# t = get_first_order_transition_counts("changed_penalties-1", "Naturalistic", "Naturalistic", 2)
# tp = get_transition_probabilities(t)
# # visualisation_method_2(tp)
#
# test = get_second_order_transition_counts("changed_penalties-1", "Naturalistic", "Naturalistic", 2)
# testp = get_transition_probabilities(test)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = testp.nonzero()
# ax.scatter(x, y, z)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

#visualise_first_order_transitions(tp)
# x = True