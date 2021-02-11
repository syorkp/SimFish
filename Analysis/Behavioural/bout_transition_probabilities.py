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


def get_previous_actions(data, a_num=0):
    previous_step_actions = []
    for index, action in enumerate(data["behavioural choice"]):
        if index == 0:
            pass
        else:
            if action == a_num:
                previous_step_actions.append(data["behavioural choice"][index-1])
    return previous_step_actions


def get_first_order_transition_probabilities(transition_counts):
    transition_probabilities = {"0": {},
                         "1": {},
                         "2": {},
                         "3": {},
                         "4": {},
                         "5": {},
                         "6": {},
                         "7": {},
                         "8": {},
                         "9": {},
                         }
    for key in transition_counts.keys():
        n = len(transition_counts[key])
        options = set(transition_counts[key])
        for action in options:
            transition_probabilities[key][action] = transition_counts[key].count(action)/n
    return transition_probabilities


def get_first_order_transition_counts(p1, p2, p3, n):
    transition_counts = {"0": [],
                         "1": [],
                         "2": [],
                         "3": [],
                         "4": [],
                         "5": [],
                         "6": [],
                         "7": [],
                         "8": [],
                         "9": [],
                         }
    for file_index in range(1, n+1):
        data = load_data("larger_network-1", "Naturalistic", f"Naturalistic-{file_index}")
        for action in range(0, 10):
            transition_counts[str(action)] = transition_counts[str(action)] + get_previous_actions(data, a_num=action)
    return transition_counts


def create_transiton_probabilities_matrix(transition_probabilities):
    matrix_tran = np.zeros((10, 10))
    for i, key in enumerate(transition_probabilities.keys()):
        for j, key2 in enumerate(transition_probabilities[key].keys()):
            matrix_tran[i, j] = transition_probabilities[key][key2]
    return matrix_tran


def visualise_first_order_transitions(transition_probabilities):
    matrix_tran = create_transiton_probabilities_matrix(transition_probabilities)
    matrix_tran = matrix_tran[:4, :4]
    mc = MarkovChain(matrix_tran, [str(i) for i in range(0, 4)])
    mc.draw()


def visualisation_method_2(transition_probabilities):
    matrix_tran = create_transiton_probabilities_matrix(transition_probabilities)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    plt.style.use(['ggplot'])
    plt.ylabel('From State')
    plt.xlabel('To State')
    mymap = plt.get_cmap("RdYlGn")
    mymap = plt.get_cmap("Reds")
    # mymap = plt.get_cmap("Greys")
    normalize = mpl.colors.LogNorm(vmin=0.0001, vmax=1)

    matrix_size = matrix_tran.shape[0]
    square_size = 1.0 / matrix_size

    diagonal = matrix_tran.diagonal()
    # colors = []

    ax.set_xticklabels(range(0, matrix_size))
    ax.set_yticklabels(range(0, matrix_size))
    ax.xaxis.set_ticks(np.arange(0 + 0.5 * square_size, 1 + 0.5 * square_size, square_size))
    ax.yaxis.set_ticks(np.arange(0 + 0.5 * square_size, 1 + 0.5 * square_size, square_size))

    # iterate over all elements of the matrix

    for i in range(0, matrix_size):
        for j in range(0, matrix_size):
            if matrix_tran[i, j] > 0:
                rect_size = np.sqrt(matrix_tran[i, j]) * square_size
            else:
                rect_size = 0

            dx = 0.5 * (square_size - rect_size)
            dy = 0.5 * (square_size - rect_size)
            p = patches.Rectangle(
                (i * square_size + dx, j * square_size + dy),
                rect_size,
                rect_size,
                fill=True,
                color=mymap(normalize(matrix_tran[i, j]))
            )
            ax.add_patch(p)

    cbax = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    cb = mpl.colorbar.ColorbarBase(cbax, cmap=mymap, norm=normalize, orientation='vertical')
    cb.set_label("Transition Probability", rotation=270, labelpad=15)

    plt.show(block=True)
    plt.interactive(False)


t = get_first_order_transition_counts("larger_network-1", "Naturalistic", "Naturalistic-", 4)
tp = get_first_order_transition_probabilities(t)
visualisation_method_2(tp)


#visualise_first_order_transitions(tp)
x = True