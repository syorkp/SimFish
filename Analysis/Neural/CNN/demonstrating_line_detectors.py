
"""Script to demonstrate that line detectors exist - bright points more likely to exist next to each other than be
randomly distributed."""

import matplotlib.pyplot as plt
import numpy as np

from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases


def get_present_distribution(image, threshold, colour_index=1):
    image = image[:, colour_index, :]
    present = image > threshold
    n_units = image.shape[-1]

    distances_between_present = []

    for unit in range(n_units):
        current_distance = 0
        for i, v in enumerate(present[:, unit]):
            if i == 0:
                pass
            if v and present[i-1, unit]:
                distances_between_present.append(current_distance)
                current_distance = 0
            else:
                current_distance += 1

    plt.hist(distances_between_present, bins=max(distances_between_present))
    plt.show()


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_efference_copy=True)
    k_l, b = get_conv_weights_and_biases(params, left=True)
    k_r, b = get_conv_weights_and_biases(params, left=False)
    get_present_distribution(np.concatenate((k_l[0], k_r[0]), axis=1), threshold=0.05, colour_index=1)

