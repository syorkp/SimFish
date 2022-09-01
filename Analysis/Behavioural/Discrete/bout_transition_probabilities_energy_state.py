"""Using plot creation from other script, displays how bout transition probabilities change according to energy state groups."""

import matplotlib.pyplot as plt
import numpy as np

from Analysis.Behavioural.Tools.BehavLabels.extract_capture_sequences import get_capture_sequences_with_energy_state
from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import get_exploration_sequences_with_energy_state
from Analysis.Behavioural.VisTools.get_action_name import get_action_name
from Analysis.Behavioural.Discrete.bout_transition_probabilities import visualisation_method_2, \
    compute_transition_probabilities, get_first_order_transition_counts_from_sequences


# TODO: Decorrelate for recent capture (where more likely to be prey nearby).

def group_sequences_by_energy_cutoffs(capture_sequences, energy_states, c, difference):
    new_capture_sequences = []
    new_energy_states = []
    for i, e in enumerate(energy_states):
        m = np.mean(e)
        if m > c and m < c + difference:
            new_capture_sequences.append(capture_sequences[i])
            new_energy_states.append(energy_states[i])
    return new_capture_sequences, new_energy_states


def display_energy_state_grouped_transition_probabilities(model_name, assay_group, assay_name, n=10,
                                                          energy_state_groups=5):
    energy_state_groups_cutoffs = np.linspace(0, 1, energy_state_groups, endpoint=False)
    difference = 1/energy_state_groups

    capture_sequences, energy_states_cs = get_capture_sequences_with_energy_state(model_name, assay_group, assay_name, n)
    exploration_sequences, energy_states_exp = get_exploration_sequences_with_energy_state(model_name, assay_group, assay_name, n)
    for c in energy_state_groups_cutoffs:
        # Capture sequences
        # new_capture_sequences, new_energy_states = group_sequences_by_energy_cutoffs(capture_sequences, energy_states_cs, c, difference)
        # transition_counts = get_first_order_transition_counts_from_sequences(new_capture_sequences)
        # tp = compute_transition_probabilities(transition_counts)
        # visualisation_method_2(tp)

        # Exploration sequences
        new_exploration_sequences, new_energy_states = group_sequences_by_energy_cutoffs(exploration_sequences, energy_states_exp, c, difference)
        transition_counts = get_first_order_transition_counts_from_sequences(new_exploration_sequences)
        tp = compute_transition_probabilities(transition_counts)
        visualisation_method_2(tp, f"Exploration Sequences {round(c, 1)}-{round(c + difference, 1)}", save_figure=True)


def plot_energy_state_grouped_action_usage_from_data(sequences, energy_states, energy_state_groups=5, save_location=None):
    energy_state_groups_cutoffs = np.linspace(0, 1, energy_state_groups, endpoint=False)
    difference = 1/energy_state_groups
    energy_state_groups_middle = np.linspace(difference/2, 1-difference/2, energy_state_groups, endpoint=True)

    actions_present = np.array([])

    action_proportions = np.zeros((10, energy_state_groups))
    for i, c in enumerate(energy_state_groups_cutoffs):
        new_sequences, new_energy_states = group_sequences_by_energy_cutoffs(sequences, energy_states, c, difference)
        sequences_flattened = np.concatenate(new_sequences)

        unique, counts = np.unique(sequences_flattened, return_counts=True)
        actions_present = np.concatenate((actions_present, unique))
        x = list(zip(unique, counts))
        for a in x:
            action_proportions[a[0], i] = a[1]/len(sequences_flattened)

    actions_present = list(set(actions_present))

    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]

    fig, ax = plt.subplots(figsize=(10, 10))
    for a in actions_present:
        ax.plot(np.linspace(0, 1, energy_state_groups), action_proportions[int(a)], c=color_set[int(a)])
    plt.xticks()
    ax.legend([get_action_name(int(a)) for a in actions_present])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    for a in reversed(list(actions_present)):
        total = sum([action_proportions[int(i)] for i in range(int(a+1))])
        ax.bar(energy_state_groups_middle, total, color=color_set[int(a)], width=0.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xticks(energy_state_groups_middle, ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"], fontsize=20)
    plt.xlabel("Energy state group", fontsize=30)
    plt.ylabel("Proportion of Actions", fontsize=30)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    # plt.legend([get_action_name(int(a)) for a in reversed(actions_present)])
    plt.savefig(save_location)

    plt.show()

def plot_energy_state_grouped_action_usage(model_name, assay_group, assay_name, n=10,
                                                          energy_state_groups=5):
    energy_state_groups_cutoffs = np.linspace(0, 1, energy_state_groups, endpoint=False)
    difference = 1/energy_state_groups
    energy_state_groups_middle = np.linspace(difference/2, 1-difference/2, energy_state_groups, endpoint=True)

    capture_sequences, energy_states_cs = get_capture_sequences_with_energy_state(model_name, assay_group, assay_name, n)
    exploration_sequences, energy_states_exp = get_exploration_sequences_with_energy_state(model_name, assay_group, assay_name, n)
    actions_present = np.array([])
    data = []
    total_actions = []

    action_proportions = np.zeros((10, energy_state_groups))
    for i, c in enumerate(energy_state_groups_cutoffs):
        # Capture sequences
        # new_capture_sequences, new_energy_states = group_sequences_by_energy_cutoffs(capture_sequences, energy_states_cs, c, difference)
        # transition_counts = get_first_order_transition_counts_from_sequences(new_capture_sequences)
        # tp = compute_transition_probabilities(transition_counts)
        # visualisation_method_2(tp)

        # Exploration sequences
        new_exploration_sequences, new_energy_states = group_sequences_by_energy_cutoffs(exploration_sequences, energy_states_exp, c, difference)
        exploration_sequences_flattened = np.concatenate(new_exploration_sequences)

        unique, counts = np.unique(exploration_sequences_flattened, return_counts=True)
        actions_present = np.concatenate((actions_present, unique))
        x = list(zip(unique, counts))
        for a in x:
            action_proportions[a[0], i] = a[1]/len(exploration_sequences_flattened)

    actions_present = list(set(actions_present))

    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]

    fig, ax = plt.subplots(figsize=(10, 10))
    for a in actions_present:
        ax.plot(np.linspace(0, 1, energy_state_groups), action_proportions[int(a)], c=color_set[int(a)])
    plt.xticks()
    ax.legend([get_action_name(int(a)) for a in actions_present])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    for a in reversed(list(actions_present)):
        total = sum([action_proportions[int(i)] for i in range(int(a+1))])
        ax.bar(energy_state_groups_middle, total, color=color_set[int(a)], width=0.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xticks(energy_state_groups_middle, ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"], fontsize=20)
    plt.xlabel("Energy state group", fontsize=30)
    plt.ylabel("Proportion of Actions", fontsize=30)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    # plt.legend([get_action_name(int(a)) for a in reversed(actions_present)])
    plt.savefig("../../Figures/Panels/Panel-5/ES-Action.png")

    plt.show()


if __name__ == "__main__":
    # display_energy_state_grouped_transition_probabilities("dqn_scaffold_18-1", f"Behavioural-Data-Free", "Naturalistic", 20, energy_state_groups=5)
    plot_energy_state_grouped_action_usage("dqn_scaffold_18-1", f"Behavioural-Data-Free", "Naturalistic", 20,
                                           energy_state_groups=5)
