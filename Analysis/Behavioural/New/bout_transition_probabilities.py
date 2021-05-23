import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import seaborn as sns
import numpy as np
import pandas as pd

import transitionMatrix as tm
from transitionMatrix import source_path
from transitionMatrix.estimators import cohort_estimator as es
# from Analysis.Behavioural.Legacy.MarkovChain.markovchain import MarkovChain


from Analysis.load_data import load_data
from Analysis.Behavioural.New.extract_event_action_sequence import get_escape_sequences, get_capture_sequences
from Analysis.Behavioural.New.turning_analysis import get_free_swimming_sequences


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


def get_fourth_order_transition_counts(p1, p2, p3, n):
    transition_counts = np.zeros((10, 10, 10, 10, 10))
    for file_index in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{file_index}")
        for i, a in enumerate(data["behavioural choice"]):
            if i < 4:
                pass
            else:
                transition_counts[data["behavioural choice"][i-4]][data["behavioural choice"][i-3]][data["behavioural choice"][i-2]][data["behavioural choice"][i-1]][a] += 1
    return transition_counts


def get_fifth_order_transition_counts(p1, p2, p3, n):
    transition_counts = np.zeros((10, 10, 10, 10, 10, 10))
    for file_index in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{file_index}")
        for i, a in enumerate(data["behavioural choice"]):
            if i < 5:
                pass
            else:
                transition_counts[data["behavioural choice"][i-5]][data["behavioural choice"][i-4]][data["behavioural choice"][i-3]][data["behavioural choice"][i-2]][data["behavioural choice"][i-1]][a] += 1
    return transition_counts


def create_third_order_transition_count_matrix(actions):
    transition_counts = np.zeros((10, 10, 10, 10))
    for i, a in enumerate(actions):
        if i == 0 or i == 1 or i == 2:
            pass
        else:
            transition_counts[actions[i - 3]][actions[i - 2]][actions[i - 1]][a] += 1
    return transition_counts


def get_first_order_transition_counts_from_sequences(sequences):
    transition_counts = np.zeros((10, 10))
    for sequence in sequences:
        for i, a in enumerate(sequence):
            if i == 0:
                pass
            else:
                transition_counts[sequence[i-1]][a] += 1
    return transition_counts


def get_third_order_transition_counts_from_sequences(sequences):
    transition_counts = np.zeros((10, 10, 10, 10))
    for sequence in sequences:
        for i, a in enumerate(sequence):
            if i == 0 or i == 1 or i == 2:
                pass
            else:
                transition_counts[sequence[i-3]][sequence[i-2]][sequence[i-1]][a] += 1
    return transition_counts


def get_fourth_order_transition_counts_from_sequences(sequences):
    transition_counts = np.zeros((10, 10, 10, 10, 10))
    for sequence in sequences:
        for i, a in enumerate(sequence):
            if i == 0 or i == 1 or i == 2 or i == 3:
                pass
            else:
                transition_counts[sequence[i-4]][sequence[i-3]][sequence[i-2]][sequence[i-1]][a] += 1
    return transition_counts

def get_fifth_order_transition_counts_from_sequences(sequences):
    transition_counts = np.zeros((10, 10, 10, 10, 10, 19))
    for sequence in sequences:
        for i, a in enumerate(sequence):
            if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
                pass
            else:
                transition_counts[sequence[i-5]][sequence[i-4]][sequence[i-3]][sequence[i-2]][sequence[i-1]][a] += 1
    return transition_counts


def get_sequence_from_index(index):
    l = index % 10
    k = round(index/10)
    j = round(index/100)
    i = round(index/1000)
    return str(i) + str(j) + str(k) + str(l)


def compute_transition_probabilities(transition_counts):
    """Returns transition probabilities as a 1D vector. Use above function to get the sequence from the 1D index"""
    transition_probabilities = transition_counts/np.sum(transition_counts)
    return transition_probabilities.ravel()


# def visualise_first_order_transitions(transition_probabilities):
#     matrix_tran = transition_probabilities[:4, :4]
#     mc = MarkovChain(matrix_tran, [str(i) for i in range(0, 4)])
#     mc.draw()


def get_action_name(action_num):
    if action_num == 0:
        action_name = "Slow2"
    elif action_num == 1:
        action_name = "RT Right"
    elif action_num == 2:
        action_name = "RT Left"
    elif action_num == 3:
        action_name = "sCS"
    elif action_num == 4:
        action_name = "JT Right"
    elif action_num == 5:
        action_name = "JT Left"
    elif action_num == 6:
        action_name = "Rest"
    elif action_num == 7:
        action_name = "SLC Right"
    elif action_num == 8:
        action_name = "SLC Left"
    elif action_num == 9:
        action_name = "AS"
    else:
        action_name = "None"
    return action_name


def visualisation_method_2(transition_probabilities):
    sns.set()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    transition_probabilities = transition_probabilities.reshape((10, 10))

    plt.style.use(['ggplot'])
    plt.ylabel('Bout 1', fontsize=15)
    plt.xlabel('Bout 2', fontsize=15)
    mymap = plt.get_cmap("YlOrRd")
    # mymap = plt.get_cmap("Reds")
    # mymap = plt.get_cmap("Greys")
    normalize = mpl.colors.LogNorm(vmin=0.0001, vmax=1)

    matrix_size = transition_probabilities.shape[0]
    square_size = 1.0 / matrix_size

    diagonal = transition_probabilities.diagonal()
    # colors = []

    ax.set_xticklabels([get_action_name(i) for i in range(10)])
    ax.set_yticklabels([get_action_name(i) for i in range(10)])
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

    # cbax = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    # cb = mpl.colorbar.ColorbarBase(cbax, cmap=mymap, norm=normalize, orientation='vertical')
    # cb.set_label("Transition Probability", rotation=270, labelpad=15)
    # plt.ylim(0, 0.4)
    # plt.xlim(0, 0.4)
    plt.show(block=True)
    plt.interactive(False)


def get_sequences_first_order():
    sequences = []
    a, b = "0", "0"
    sequences.append(a + b)
    for i in range(1, 100):
        b = str(int(b) + 1)
        if i % 10 == 0:
            b = "0"
            a = str(int(a) + 1)
        sequences.append(a + b)
    return sequences


def get_sequences_second_order():
    sequences = []
    a, b, c = "0", "0", "0"
    sequences.append(a + b + c)
    for i in range(1, 1000):
        c = str(int(c) + 1)
        if i % 10 == 0:
            c = "0"
            b = str(int(b) + 1)
        if i % 100 == 0:
            b = "0"
            a = str(int(a) + 1)
        sequences.append(a + b + c)
    return sequences


def get_sequences_third_order():
    sequences = []
    a, b, c, d = "0", "0", "0", "0"
    sequences.append(a + b + c + d)
    for i in range(1, 10000):
        d = str(int(d) + 1)
        if i % 10 == 0:
            d = "0"
            c = str(int(c) + 1)
        if i % 100 == 0:
            c = "0"
            b = str(int(b) + 1)
        if i % 1000 == 0:
            b = "0"
            a = str(int(a) + 1)
        sequences.append(a + b + c + d)
    return sequences


def get_sequences_fourth_order():
    sequences = []
    a, b, c, d, e = "0", "0", "0", "0", "0"
    sequences.append(a + b + c + d + e)
    for i in range(1, 100000):
        e = str(int(e) + 1)
        if i % 10 == 0:
            e = "0"
            d = str(int(d) + 1)
        if i % 100 == 0:
            d = "0"
            c = str(int(c) + 1)
        if i % 1000 == 0:
            c = "0"
            b = str(int(b) + 1)
        if i % 10000 == 0:
            b = "0"
            a = str(int(a) + 1)
        sequences.append(a + b + c + d + e)
    return sequences


def get_sequences_fifth_order():
    sequences = []
    a, b, c, d, e, f = "0", "0", "0", "0", "0", "0"
    sequences.append(a + b + c + d + e + f)
    for i in range(1, 1000000):
        f = str(int(f) + 1)
        if i % 10 == 0:
            f = "0"
            e = str(int(e) + 1)
        if i % 100 == 0:
            e = "0"
            d = str(int(d) + 1)
        if i % 1000 == 0:
            d = "0"
            c = str(int(c) + 1)
        if i % 10000 == 0:
            c = "0"
            b = str(int(b) + 1)
        if i % 100000 == 0:
            b = "0"
            a = str(int(a) + 1)
        sequences.append(a + b + c + d + e + f)
    return sequences


def get_transition_probabilities(model, assay_name, assay_id, number_of_files, order):
    if order == 1:
        t_counts = get_first_order_transition_counts(model, assay_name, assay_id, number_of_files)
    elif order == 2:
        t_counts = get_second_order_transition_counts(model, assay_name, assay_id, number_of_files)
    elif order == 3:
        t_counts = get_third_order_transition_counts(model, assay_name, assay_id, number_of_files)
    elif order == 4:
        t_counts = get_fourth_order_transition_counts(model, assay_name, assay_id, number_of_files)
    elif order == 5:
        t_counts = get_fifth_order_transition_counts(model, assay_name, assay_id, number_of_files)
    else:
        print("Incorrect order given")
        t_counts = None
    t_probabilities = compute_transition_probabilities(t_counts)
    return t_probabilities


def get_modal_sequences(transition_probabilities, order=3, number=10):
    if order == 1:
        sequences = get_sequences_first_order()
    elif order == 2:
        sequences = get_sequences_second_order()
    elif order == 3:
        sequences = get_sequences_third_order()
    elif order == 4:
        sequences = get_sequences_fourth_order()
    elif order == 5:
        sequences = get_sequences_fifth_order()
    else:
        print("Incorrect order given")
        sequences = None

    ordered_probabilities, ordered_sequences = zip(*sorted(zip(transition_probabilities, sequences)))
    ordered_probabilities = list(ordered_probabilities)
    ordered_sequences = list(ordered_sequences)
    ordered_probabilities.reverse()
    ordered_sequences.reverse()
    ordered_sequences = ordered_sequences[:number]
    for i, seq in enumerate(ordered_sequences):
        ordered_sequences[i] = [int(l) for l in seq]
    return ordered_sequences

free_swimiming = []
for i in range(1, 11):
    for j in range(1, 4):
        cs = get_free_swimming_sequences(load_data("new_differential_prey_ref-3", f"Behavioural-Data-Free-{j}", f"Naturalistic-{i}"))
        free_swimiming += cs
transition_counts = get_first_order_transition_counts_from_sequences(free_swimiming)
tp = compute_transition_probabilities(transition_counts)
visualisation_method_2(tp)
x = True
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