import copy

import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases


def compute_optimal_filter_input(kernels, biases, input_size, output_dim=2):
    """From reverse, constructs the optimal input."""

    for final_unit in range(kernels[-1].shape[2]):
        final_unit_kernel = kernels[-1][:, :, final_unit]
        output = np.ones((output_dim))

        stride = 1
        kernel_size = kernels[-1].shape[0]
        previous_output_size = 5

        stride_points = [i for i in range(0, previous_output_size, stride) if i + kernel_size <= previous_output_size]
        previous_layer_n_units = kernels[-2].shape[2]
        next_step_output = np.zeros((previous_output_size, previous_layer_n_units))

        for i, s in enumerate(stride_points):
            for c in range(previous_layer_n_units):
                relevant_kernel = final_unit_kernel[:, c]
                reversed_kernel = output[i] / relevant_kernel
                next_step_output[s: s+kernel_size, c] += reversed_kernel

        x = True


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k, b = get_conv_weights_and_biases(params, left=True)
    compute_optimal_filter_input(k, b, 100)
