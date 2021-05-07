import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Analysis.Behavioural.bout_transition_probabilities import get_modal_sequences, get_transition_probabilities, get_third_order_transition_counts, compute_transition_probabilities
from Analysis.Behavioural.extract_event_action_sequence import get_escape_sequences, get_capture_sequences, create_density_matrix


def display_sequences(sequences):
    plot_dim = len(sequences[0])
    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'y', "k", "m", "m", "k"]
    plt.figure(figsize=(5, 15))
    for i, seq in enumerate(reversed(sequences)):
        for j, a in enumerate(reversed(seq)):
            j = plot_dim - j
            plt.fill_between((j, j + 1), i, i + 1, color=color_set[a])
    plt.axis("scaled")
    plt.show()


def display_all_sequences_capture(sequences):
    sequences.sort(key=len)
    plot_dim = max([len(seq) for seq in sequences])

    color_set = ['b', 'g', 'g', 'r', 'y', 'y', "k", "m", "m", "k"]
    plt.figure(figsize=(5, 15))
    for i, seq in enumerate(sequences):
        for j, a in enumerate(reversed(seq)):
            j = plot_dim - j
            plt.fill_between((j, j+1), i, i+1, color=color_set[a])
    plt.axis("scaled")
    plt.show()


def display_all_sequences_escape(sequences):
    sequences.sort(key=len)
    plot_dim = max([len(seq) for seq in sequences])
    color_set = ['b', 'g', 'g', 'r', 'y', 'y', "k", "m", "m", "b"]
    plt.figure(figsize=(5, 15))
    for i, seq in enumerate(sequences):
        for j, a in enumerate(reversed(seq)):
            j = plot_dim - j
            plt.fill_between((j, j+1), i, i+1, color=color_set[a])
    plt.axis("scaled")
    plt.show()


def display_average_sequence(sequences):
    plot_dim = max([len(seq) for seq in sequences])
    modal_sequence_1 = []
    modal_sequence_2 = []
    modal_sequence_3 = []
    color_set = ['b', 'g', 'g', 'r', 'y', 'y', "k", "m", "m"]
    for index in range(max([len(seq) for seq in sequences])-1):
        all_actions_at_index = [seq[len(seq) - index - 1] for seq in sequences if len(seq) > index+1]
        modal_1 = max(set(all_actions_at_index), key=all_actions_at_index.count)
        remaining = list(set(all_actions_at_index))
        remaining.remove(modal_1)
        modal_2 = max(set(remaining), key=all_actions_at_index.count)
        # remaining.remove(modal_2)
        # modal_3 = max(set(remaining), key=all_actions_at_index.count)
        modal_sequence_1.append(modal_1)
        modal_sequence_2.append(modal_2)
        # modal_sequence_3.append(modal_3)

    for i, a in enumerate(reversed((modal_sequence_1))):
        i = plot_dim - i
        plt.fill_between((i, i + 1), 1, color=color_set[a])
    plt.axis("scaled")
    plt.show()

    for i, a in enumerate(reversed((modal_sequence_2))):
        i = plot_dim - i
        plt.fill_between((i, i + 1), 1, color=color_set[a])
    plt.axis("scaled")
    plt.show()
    #
    # for i, a in enumerate(reversed((modal_sequence_3))):
    #     i = plot_dim - i
    #     plt.fill_between((i, i + 1), 1, color=color_set[a])
    # plt.axis("scaled")
    # plt.show()


def display_relative_probabilities(transition_probabilities):
    transition_probabilities = transition_probabilities.reshape((10, 10))
    x = False

# # capture_sequences = get_capture_sequences("even_prey_ref-3", "Behavioural-Data-Free", "Prey", 1)
# # display_all_sequences_capture(capture_sequences[:70])
# escape_sequences = get_escape_sequences("even_prey_ref-3", "Behavioural-Data-Free", "Predator", 1)
# # display_all_sequences_escape(escape_sequences[:70])
#
# escape_sequences = escape_sequences + get_escape_sequences("even_prey_ref-4", "Behavioural-Data-Free", "Predator", 1)
# # display_all_sequences_escape(escape_sequences[:70])
# escape_sequences = escape_sequences + get_escape_sequences("even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 1)
# display_all_sequences_escape(escape_sequences[:60])

# cs = get_capture_sequences("even_prey_ref-7", "Ablation-Test-Predator-Only", "Prey-Only-Ablated-100", 3)
# display_all_sequences_capture(cs)
#
# cs = get_capture_sequences("even_prey_ref-7", "Ablation-Test-Predator-Only", "Prey-Only", 3)
# display_all_sequences_capture(cs)

# tp = get_transition_probabilities("changed_penalties-1", "Naturalistic", "Naturalistic", 2, order=1)
# display_relative_probabilities(tp)
# ms = get_modal_sequences(tp, order=5, number=40)
# display_sequences(ms)

