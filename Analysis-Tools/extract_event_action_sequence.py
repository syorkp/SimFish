import json
import numpy as np
from matplotlib import pyplot as plt
from load_data import load_data

"""
Should, given two indexes, give the full action sequence of the fish within.
"""


def extract_predator_action_sequences(data):
    """Returns all action sequences that occur while a predator is present"""


def extract_consumption_action_sequences(data, n=10):
    """Returns all action sequences that occur n steps before consumption"""


data = load_data("simple_actions", "All-Features", "conditional_prey_and_predators-1")

consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]