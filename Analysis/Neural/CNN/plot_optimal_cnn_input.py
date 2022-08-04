import copy

import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases


def get_previous_layer(kernel, output, stride, previous_output_size, previous_layer_n_units):
    kernel_size = kernel.shape[0]
    stride_points = [i for i in range(0, previous_output_size, stride) if i + kernel_size <= previous_output_size]

    next_step_output = np.zeros((previous_output_size, previous_layer_n_units))

    for i, s in enumerate(stride_points):
        for c in range(previous_layer_n_units):
            relevant_kernel = kernel[:, c]
            reversed_kernel = output[i] / relevant_kernel
            next_step_output[s: s + kernel_size, c] += reversed_kernel

    return next_step_output

def compute_optimal_filter_input(kernels, biases, input_size, output_dim=2):
    """From reverse, constructs the optimal input."""

    for final_unit in range(kernels[-1].shape[2]):
        output = np.ones((output_dim))
        final_unit_kernel = kernels[-1][:, :, final_unit]

        for i, kernel in enumerate(reversed(kernels)):
            if i == 3:
                stride = 4
                previous_layer_size = 100
            elif i == 2:
                stride = 2
                previous_layer_size = 22
            elif i == 1:
                stride = 1
                previous_layer_size = 8
            else:
                stride = 1
                previous_layer_size = 5

            if i == 3:
                previous_layer_n_units = 3
            else:
                previous_layer_n_units = kernels[-(i+2)].shape[2]

            # TODO: Need to get the correct kernel - am doing it for each final layer individually, just need to do the
            #  other layers differently, as there are more axes.
            output = get_previous_layer(kernel, output, stride, previous_layer_size, previous_layer_n_units)
            # TODO: normalise at each point before applying bias.
            output += biases[-(i+1)][final_unit]
            output = np.sum(output, axis=1)


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k, b = get_conv_weights_and_biases(params, left=True)
    compute_optimal_filter_input(k, b, 100)
