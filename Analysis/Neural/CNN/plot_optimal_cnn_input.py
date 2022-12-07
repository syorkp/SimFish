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

    for unit in range(kernel.shape[-1]):
        for i, s in enumerate(stride_points):
            for c in range(previous_layer_n_units):
                relevant_kernel = kernel[:, c, unit]
                reversed_kernel = output[i] / relevant_kernel
                next_step_output[s: s + kernel_size, c] += reversed_kernel

    return next_step_output


def compute_optimal_filter_input(kernels, biases, input_size, output_dim=2, activation_function="relu"):
    """From reverse, constructs the optimal input."""
    num_final_kernels = kernels[-1].shape[2]

    optimal_output_compiled = np.zeros((num_final_kernels, input_size, 3))
    for final_unit in range(num_final_kernels):
        output = np.ones((output_dim))
        final_unit_kernel = kernels[-1][:, :, final_unit:final_unit+1]

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
                kernel = final_unit_kernel

            if i == 3:
                previous_layer_n_units = 3
                chosen_bias = 0
            else:
                previous_layer_n_units = kernels[-(i+2)].shape[2]
                chosen_bias = np.expand_dims(biases[-(i+2)], 0)

            output = get_previous_layer(kernel, output, stride, previous_layer_size, previous_layer_n_units)

            # Normalise output.
            output = output / max(abs(np.max(output)), abs(np.min(output)))
            # output = np.clip(output, -0.2, 0.2) * 5  # TODO: Decide whether is valid.
            # output += chosen_bias
            # Collapse along constituent kernels.
            if i != 3:
                output = np.sum(output, axis=1)

            # if activation_function == "relu":
            #     output = np.maximum(output, 0)

        output = np.expand_dims(output, 0)
        optimal_output_compiled[final_unit, :, :] = output

    # Change to be RGB
    optimal_output_compiled = np.concatenate((optimal_output_compiled[:, :, 0:1], optimal_output_compiled[:, :, 2:3],
                                              optimal_output_compiled[:, :, 1:2]), axis=2)
    # Remove R2 channel
    optimal_output_compiled[:, :, 1] *= 0

    # Invert image
    plt.imshow(optimal_output_compiled)
    plt.show()


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k, b = get_conv_weights_and_biases(params, left=False)
    compute_optimal_filter_input(k, b, 100)


    for i, layer in enumerate(reversed(relevant_layers)):
        input_weights = np.sum(layer, axis=0)
        ongoing_weights = np.sum(ongoing_weights, 1)
