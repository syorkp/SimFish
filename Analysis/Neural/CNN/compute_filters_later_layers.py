import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases
from Analysis.Neural.CNN.plot_cnn_filters import display_cnn_filters


def get_filter_shapes(kernels):
    """From first kernel, constructs shape of later kernels through computation, preserving the original kernel size.
    Works by summing input filters, weighted by weights between.
    Returns shape of: (kernel_size_first_layer, channels_first_layer, n_units_final_layer)
    """
    n_layers = len(kernels)
    output_shape = (kernels[0].shape[0], kernels[0].shape[1], kernels[-1].shape[-1])
    output_filters = np.zeros(output_shape)

    # Compile weights (reduce their dimensionality)
    layer_2 = kernels[-1]
    for l in range(1, n_layers):
        layer_1 = kernels[-(l+1)]
        output = np.zeros((layer_1.shape[0], layer_1.shape[1], layer_2.shape[-1]))

        for unit in range(layer_2.shape[-1]):
            final_unit = layer_2[:, :, unit]
            final_unit = np.swapaxes(final_unit, 0, 1)

            # Collapse along second kernel dimension
            final_unit = np.sum(final_unit, axis=1)

            scaled_input = layer_1 * final_unit
            resolved = np.sum(scaled_input, axis=(2))
            # resolved = np.expand_dims(resolved, 0)
            output[:, :, unit] = resolved
        layer_2 = output

    return output

    # output = np.swapaxes(output, 1, 2)
    # # Swap so kernel on bottom.
    # output = np.swapaxes(output, 0, 1)
    # output = np.concatenate((output[:, :, 0:1], output[:, :, 2:3], output[:, :, 1:2]), axis=2)
    # output[:, :, 1] *= 0
    # plt.imshow(output)
    # plt.show()


def compute_filter_shapes_rnn(kernels_l, kernels_r, rnn_in_weights):
    final_filter_shapes_l = get_filter_shapes(kernels_l)
    final_filter_shapes_r = get_filter_shapes(kernels_r)

    output_size = final_filter_shapes_l.shape[-1]*2

    all_units = np.zeros((2, 512, 16, 3))

    for unit in range(rnn_in_weights.shape[-1]):
        rnn_in_weights_l = rnn_in_weights[:output_size, unit]
        rnn_in_weights_r = rnn_in_weights[output_size:output_size*2, unit]

        # Note that output of the flatten operation underlying formation of (2, 64) to 128 involves stacking 64-64,
        final_filter_shapes_l_current = final_filter_shapes_l * rnn_in_weights_l[:int(output_size / 2)]
        final_filter_shapes_r_current = final_filter_shapes_r * rnn_in_weights_r[:int(output_size / 2)]

        final_filter_shapes_l_current = np.sum(final_filter_shapes_l_current, axis=2)
        final_filter_shapes_r_current = np.sum(final_filter_shapes_r_current, axis=2)

        all_units[0, unit, :, :] = final_filter_shapes_l_current
        all_units[1, unit, :, :] = final_filter_shapes_r_current

    return all_units
    plt.figure(figsize=(8, 30))
    plt.imshow(all_units[0])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 30))
    plt.imshow(all_units[1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k_l, b_l = get_conv_weights_and_biases(params, left=True)
    k_r, b_r = get_conv_weights_and_biases(params, left=False)
    # get_filter_shapes_poc(k[0], k[1])
    # filters = get_filter_shapes(k)
    # display_cnn_filters([filters], True, True, mask_red=True, normalisation_mode="rescale")
    rnn_filter_shapes = compute_filter_shapes_rnn(k_l, k_r, params["main_rnn_in/kernel:0"])
    rnn_filter_shapes_l = np.swapaxes(rnn_filter_shapes[0, :40], 0, 2)
    rnn_filter_shapes_r = np.swapaxes(rnn_filter_shapes[1, :40], 0, 2)
    rnn_filter_shapes_l = np.swapaxes(rnn_filter_shapes_l, 0, 1)
    rnn_filter_shapes_r = np.swapaxes(rnn_filter_shapes_r, 0, 1)
    display_cnn_filters([rnn_filter_shapes_l], True, True, mask_red=True, normalisation_mode="rescale")
    display_cnn_filters([rnn_filter_shapes_r], True, True, mask_red=True, normalisation_mode="rescale")
