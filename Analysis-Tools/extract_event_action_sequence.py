import json
import numpy as np
from matplotlib import pyplot as plt
from load_data import load_data

"""
Should, given two indexes, give the full action sequence of the fish within.
"""


def extract_predator_action_sequences(data, n=10):
    """Returns all action sequences that occur while a predator is present"""
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    action_sequences = []
    while len(predator_timestamps) > 0:
        index = predator_timestamps.pop(0)
        predator_timestamps = [i for i in range(index-n, index) if i >= 0]

        # single_timestamps = [t for i, t in enumerate(predator_timestamps) if t == predator_timestamps[i-1] + 1 or i == 0]
        # predator_timestamps = predator_timestamps[len(single_timestamps):]
        # action_sequence = [data["behavioural choice"][i] for i in single_timestamps]

        action_sequence = [data["behavioural choice"][i] for i in predator_timestamps]
        action_sequences.append(action_sequence)
    return action_sequences


def extract_consumption_action_sequences(data, n=10):
    """Returns all action sequences that occur n steps before consumption"""
    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    action_sequences = []
    while len(consumption_timestamps) > 0:
        index = consumption_timestamps.pop(0)
        prey_capture_timestamps = [i for i in range(index-n, index) if i >= 0]
        action_sequence = [data["behavioural choice"][i] for i in prey_capture_timestamps]
        action_sequences.append(action_sequence)
    return action_sequences


#model_name = "realistic_all_features-2"
model_name = "basic_all_features-1"

consumption_sequence_location = "../Assay-Output/prey_capture.json"
prey_capture_sequences = []

all_capture_sequences = []
for i in range(1, 5):
    data = load_data("Prey-Capture", f"Prey_Capture-{i}", model_name)
    all_capture_sequences = all_capture_sequences + extract_consumption_action_sequences(data)

data = None
all_avoidance_sequences = []
for i in range(1, 5):
    data = load_data("Predator-Avoidance", f"Predator_Avoidance-{i}", model_name)
    all_avoidance_sequences = all_avoidance_sequences + extract_predator_action_sequences(data)

data = None
all_avoidance_only_sequences = []
for i in range(1, 5):
    data = load_data("Predator-Only-Avoidance", f"Predator_Avoidance-{i}", model_name)
    all_avoidance_only_sequences = all_avoidance_only_sequences + extract_predator_action_sequences(data)


# Create a density matrix. Heatmap is 10 ( x 6 (actions)
def create_density_matrix(sequences):
    # TODO: Actually need to "count back" for the cases when the list is shorter than 10.
    colour_density_matrix = np.zeros((6, 10), dtype=int)
    for sequence in sequences:
        for j, n in enumerate(sequence):
            colour_density_matrix[n][j] += 1
    return colour_density_matrix


dm_capture = create_density_matrix(all_capture_sequences)

dm_avoidance = create_density_matrix(all_avoidance_sequences)

dm_avoidance_only = create_density_matrix(all_avoidance_only_sequences)

x = True