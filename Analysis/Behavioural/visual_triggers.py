import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

from Analysis.load_data import load_data


"""
Tools to display the average visual input received when: A) A specific bout is chosen, B) A specific behavioural sequence is initiated.
"""

data = load_data("changed_penalties-1", "Naturalistic", "Naturalistic-1")

x = True


def average_visual_input_for_bout_sequence(p1, p2, p3, n, bout_sequence):
    ...


def average_visual_input_for_bout(p1, p2, p3, n, bout_num):
    # Get all the observation data from the required firles for the given bout number.
    # Perform some kind of average of the photons to get a new image.
    #
    return


average_visual_input_for_bout("changed_penalties-1", "Naturalistic", "Naturalistic-", 2, 5)