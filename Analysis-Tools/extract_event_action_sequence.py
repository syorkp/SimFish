import json
import numpy as np
from matplotlib import pyplot as plt
from load_data import load_data

"""
Should, given two indexes, give the full action sequence of the fish within.
"""


def extract_predator_action_sequences(data):
    """Returns all action sequences that occur while a predator is present"""
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    action_sequences = []
    while len(predator_timestamps) > 0:
        single_timestamps = [t for i, t in enumerate(predator_timestamps) if t == predator_timestamps[i-1] + 1 or i == 0]
        predator_timestamps = predator_timestamps[len(single_timestamps):]
        action_sequence = [data["behavioural choice"][i] for i in single_timestamps]
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


data = load_data("simple_actions", "All-Features", "realistic_all_features-1")
x = extract_predator_action_sequences(data)
y = extract_consumption_action_sequences(data)
a = True
