import copy

import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases


def compute_filters(kernels, biases, input_filter, activation="relu"):
    """
    Kernels have the following dimensions: (filters, channel_number, kernel_size)
    :return:
    """
    input_size = input_filter.shape[0]
    # Create new filter.

    for layer_num in range(len(kernels)):
        if layer_num == 0:
            strides = 4
        elif layer_num == 1:
            strides = 2
        else:
            strides = 1

        new_filter = np.zeros((input_size, 3, kernels[layer_num].shape[-1]))

        for unit in range(kernels[layer_num].shape[-1]):
            kernel_size = kernels[layer_num].shape[0]
            # Convolution of kernel across input
            for i in range(0, input_size, strides):
                if len(new_filter[i:i + kernel_size, 0]) == kernel_size:
                    for c in range(input_filter.shape[1]):
                        if layer_num == 0:
                            new_filter[i:i + kernel_size, c, unit] += input_filter[i:i+kernel_size, c] * kernels[layer_num][:, c, unit]
                        else:
                            input_array = input_filter[i:i+kernel_size, :, c]
                            kernel_array = kernels[layer_num][:, c:c+1, unit]
                            element_by_element_mul = input_array * kernel_array
                            new_filter[i:i + kernel_size, :, unit] += element_by_element_mul
            # Add the bias
            new_filter[:, :, unit] += biases[layer_num][unit]
            # Apply activation function
            if activation == "relu":
                new_filter = np.maximum(new_filter, 0)
            else:
                print("ERROR, activation function not recognised.")

        # Scaling for full range display
        # scaling_factor = 1/np.max(new_filter)
        # new_filter *= scaling_factor

        input_filter = new_filter

    # Normalise
    scaling_factor = 1 / np.max(new_filter)
    new_filter *= scaling_factor

    return new_filter


def display_cnn_feature_map(kernels, biases, input_size=100, activation_factor=1.0, input_image=None, display_all=False):
    """NOTE: Must supply layer weights in correct order, from first to last."""
    if input_image is None:
        input_image = np.ones((input_size, 3)) * activation_factor

    for layer_num in range(len(kernels)):
        rel_kernels = kernels[:layer_num + 1]
        rel_biases = biases[:layer_num + 1]

        computed_filter = compute_filters(rel_kernels, rel_biases, input_image)
        computed_filter = np.swapaxes(computed_filter, 1, 2)
        computed_filter = np.swapaxes(computed_filter, 0, 1)
        computed_filter[:, :, 2] *= 0

        # Swap blue location so is RGB
        computed_filter = np.concatenate((computed_filter[:, :, 0:1], computed_filter[:, :, 2:3], computed_filter[:, :, 1:2]), axis=2)

        if display_all or layer_num == 3:
            input_image = np.concatenate((input_image[:, 0:1], np.zeros((100, 1)), input_image[:, 1:2]), axis=1).astype(int)
            input_image = np.expand_dims(input_image, 0)
            input_image = np.repeat(input_image, 10, 0)
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].imshow(input_image, aspect="auto")
            axs[1].imshow(computed_filter, aspect="auto")
            axs[0].autoscale()
            plt.tight_layout()
            plt.show()


def display_cnn_feature_map2(kernels, biases, input_image, display_all=False):
    for layer_num in range(len(kernels)):
        rel_kernels = kernels[:layer_num + 1]
        rel_biases = biases[:layer_num + 1]

        computed_filter = compute_filters2(rel_kernels, rel_biases, input_image)

        if display_all or layer_num == 3:
            # Preparing input image
            input_image_p = copy.copy(input_image.astype(int))
            input_image_p = np.expand_dims(input_image_p, 0)

            # Preparing filter
            n_units = computed_filter.shape[-1]
            computed_filter = np.swapaxes(computed_filter, 0, 1)

            fig, axs = plt.subplots(n_units+1, 1)
            fig.set_size_inches(16, 16)
            axs[0].imshow(input_image_p, aspect="auto")
            for unit in range(n_units):
                axs[unit+1].imshow(computed_filter[unit:unit+1, :], aspect="auto")
                axs[unit+1].tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off
            axs[0].autoscale()

            plt.tight_layout()
            plt.show()


def compute_filters2(kernels, biases, observation, activation="relu"):
    # Create new filter.
    input_filter = observation

    for layer_num in range(len(kernels)):
        if layer_num == 0:
            strides = 4
        elif layer_num == 1:
            strides = 2
        else:
            strides = 1

        input_size = input_filter.shape[0]

        kernel_size = kernels[layer_num].shape[0]

        stride_points = [i for i in range(0, input_size, strides) if i + kernel_size <= input_size]

        num_output_units = kernels[layer_num].shape[-1]
        output = np.zeros((len(stride_points), num_output_units))

        for unit in range(num_output_units):
            # Convolution of kernel across input
            for i, s in enumerate(stride_points):
                for c in range(input_filter.shape[1]):
                    input_f = input_filter[s:s+kernel_size, c]
                    kernel_v = kernels[layer_num][:, c, unit]
                    output[i, unit] += np.sum(input_f * kernel_v)

            # Add the bias
            output[:, unit] += biases[layer_num][unit]
            # Apply activation function
            if activation == "relu":
                output = np.maximum(output, 0)
            else:
                print("ERROR, activation function not recognised.")

        # Scaling for full range display
        # scaling_factor = 1/np.max(new_filter)
        # new_filter *= scaling_factor

        input_filter = output

    # Normalise
    scaling_factor = 1 / np.max(output)
    output *= scaling_factor

    return output


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k, b = get_conv_weights_and_biases(params, left=True)
    for i in range(5):
        random_observation = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")["observation"][100+i, :, :, 0]
        # compute_optimal_filter_input(k, 100)
        # display_cnn_filters_pure(k)
        display_cnn_feature_map2(k, b, input_image=random_observation, display_all=True)
        # display_cnn_feature_map(k, b, activation_factor=1.0, input_image=random_observation, display_all=False)

    #
    # set_observation = np.ones((100, 3))
    # set_observation[:, 0] = 2
    # set_observation[:, 1] = 18
    # set_observation[:, 2] = 18
    # set_observation[20:22, 1] = 40
    # set_observation[78:80, 1] = 40
    # display_cnn_feature_map2(k, b, input_image=set_observation, display_all=True)
    #
    # set_observation = np.ones((100, 3))
    # set_observation[:, 0] *= 2
    # set_observation[:, 1] *= 18
    # set_observation[:, 2] *= 18
    # set_observation[20:22, 1] *= 40
    # display_cnn_feature_map(k, b, activation_factor=1.0, input_image=set_observation, display_all=False)
    #
    # set_observation = np.ones((100, 3))
    # set_observation[:, 0] *= 2
    # set_observation[:, 1] *= 18
    # set_observation[:, 2] *= 18
    # set_observation[78:80, 1] *= 40
    # display_cnn_feature_map(k, b, activation_factor=1.0, input_image=set_observation, display_all=False)

