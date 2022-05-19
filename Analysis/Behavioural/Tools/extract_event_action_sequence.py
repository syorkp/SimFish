import json
import numpy as np
from matplotlib import pyplot as plt
from Analysis.load_data import load_data

"""
Should, given two indexes, give the full action sequence of the fish within.
"""


def extract_predator_action_sequences(data):
    """Returns all action sequences that occur while a predator is present"""
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    predator_sequence_timestamps = []
    current_sequence = []
    prev = 0
    while len(predator_timestamps) > 0:
        index = predator_timestamps.pop(0)
        if index == prev +1 or prev == 0:
            current_sequence.append(index)
        else:
            predator_sequence_timestamps.append(current_sequence)
            current_sequence = [index]
        prev = index
    action_sequences = []
    for sequence in predator_sequence_timestamps:
        action_sequence = [data["action"][i] for i in sequence]
        action_sequences.append(action_sequence)
    return action_sequences, predator_sequence_timestamps





def create_density_matrix(sequences):
    # TODO: Actually need to "count back" for the cases when the list is shorter than 10.
    colour_density_matrix = np.zeros((9, 10), dtype=int)
    for sequence in sequences:
        for j, n in enumerate(sequence):
            colour_density_matrix[n][j] += 1
    return colour_density_matrix


def get_escape_sequences(model_name, assay_config, assay_id, n):
    all_escape_sequences = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_escape_sequences = all_escape_sequences + extract_predator_action_sequences(data)[0]
    return all_escape_sequences

# capture_sequences = get_capture_sequences("large_all_features-1", "Naturalistic", "Naturalistic", 2)
# escape_sequences = get_escape_sequences("large_all_features-1", "Predator", "Predator", 4)
# dm_capture = create_density_matrix(capture_sequences)
# dm_avoidance = create_density_matrix(escape_sequences)

