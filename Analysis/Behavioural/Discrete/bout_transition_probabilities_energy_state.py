"""Using plot creation from other script, displays how bout transition probabilities change according to energy state groups."""

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.extract_capture_sequences import get_capture_sequences, get_capture_sequences_with_energy_state
from Analysis.Behavioural.Tools.extract_exploration_sequences import get_exploration_sequences, get_exploration_sequences_with_energy_state

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

    capture_sequences, energy_states = get_capture_sequences_with_energy_state(model_name, assay_group, assay_name, n)
    exploration_sequences, energy_states = get_exploration_sequences_with_energy_state(model_name, assay_group, assay_name, n)
    for c in energy_state_groups_cutoffs:
        # Capture sequences
        new_capture_sequences, new_energy_states = group_sequences_by_energy_cutoffs(capture_sequences, energy_states, c, difference)
        transition_counts = get_first_order_transition_counts_from_sequences(new_capture_sequences)
        tp = compute_transition_probabilities(transition_counts)
        visualisation_method_2(tp)

        # Exploration sequences
        new_exploration_sequences, new_energy_states = group_sequences_by_energy_cutoffs(exploration_sequences, energy_states, c, difference)
        transition_counts = get_first_order_transition_counts_from_sequences(new_exploration_sequences)
        tp = compute_transition_probabilities(transition_counts)
        visualisation_method_2(tp)


display_energy_state_grouped_transition_probabilities("dqn_scaffold_14-1", f"Behavioural-Data-Free", "Naturalistic", 10, energy_state_groups=5)
