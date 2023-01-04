import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data

"""
For creation of displays of action space usage across training.
"""


def display_impulse_angle_space_usage(impulses, angles, overlay_action_mask=False, overlay_bout_pdfs=False,
                                      max_impulse=None, max_angle=None):
    x = True


if __name__ == "__main__":
    d = load_data("dqn_beta-1", "Behavioural-Data-Free", "Naturalistic-1")
    display_impulse_angle_space_usage(d["impulse"], d["angle"])

