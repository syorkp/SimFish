import copy

import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases


def display_cnn_filters(layers):
    """Displays kernels of CNN filters."""
    for i, layer in enumerate(layers):
        n_units = layer.shape[-1]
        layer = np.swapaxes(layer, 0, 1)
        fig, axs = plt.subplots(int(n_units/2), 2)
        fig.set_size_inches(10, 30)
        for n in range(n_units):
            if n % 2 == 0:
                axs[int(n/2), 0].imshow(layer[:, :, n])
            else:
                axs[int(n / 2), 1].imshow(layer[:, :, n])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k, b = get_conv_weights_and_biases(params, left=True)
    display_cnn_filters(k)
