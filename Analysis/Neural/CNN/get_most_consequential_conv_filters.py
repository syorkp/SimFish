import numpy as np

from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases
from Analysis.Neural.CNN.reorder_filters import reorder_filters
from Analysis.Neural.CNN.plot_cnn_filters import display_cnn_filters


def get_most_consequential_conv_filters_for_conv(output_conv_layer, output_conv_index, target_conv_layer, kernels,
                                                 biases):
    # layer_input_sizes = [100, 22, 8, 5, 2]
    # input_sizes = [layer_input_sizes[i] for i in range(target_conv_layer, output_conv_layer+1)]
    # input_s = input_sizes[0]
    # output_s = input_sizes[-1]

    relevant_layers = kernels[target_conv_layer: output_conv_layer]
    final_unit = kernels[-1][:, :, output_conv_index:output_conv_index+1]
    ongoing_weights = np.sum(final_unit, axis=0)

    for i, layer in enumerate(reversed(relevant_layers)):
        input_weights = np.sum(layer, axis=0)
        bias = biases[-(i+2)]
        ongoing_weights = ongoing_weights * input_weights
        ongoing_weights += bias
        if i != len(relevant_layers) - 1:
            ongoing_weights = np.sum(ongoing_weights, axis=1)

    ongoing_weights = np.absolute(ongoing_weights)

    importance_for_determining_output = np.zeros(ongoing_weights.shape)
    total_input_kernels = ongoing_weights.shape[0] * ongoing_weights.shape[1]

    for i in range(total_input_kernels):
        strongest = np.unravel_index(np.argmax(ongoing_weights), ongoing_weights.shape)
        ongoing_weights[strongest] = 0
        importance_for_determining_output[strongest] = i

    return importance_for_determining_output


def get_most_consequential_conv_filters_for_rnn(target_conv_layer, rnn_index, left_kernels, right_kernels, left_biases,
                                                right_biases, rnn_in_weights, output_size=128):
    rnn_in_weights_l = rnn_in_weights[:output_size, rnn_index]
    rnn_in_weights_r = rnn_in_weights[output_size:output_size*2, rnn_index]
    num_final_filters = left_kernels[-1].shape[-1]
    importance_tally = np.zeros((2, num_final_filters, left_kernels[target_conv_layer].shape[1],
                                 left_kernels[target_conv_layer].shape[0]))
    for i in range(num_final_filters):
        l_imp = get_most_consequential_conv_filters_for_conv(3, i, target_conv_layer, left_kernels, left_biases)
        r_imp = get_most_consequential_conv_filters_for_conv(3, i, target_conv_layer, right_kernels, right_biases)
        importance_tally[0, i, :, :] = l_imp
        importance_tally[1, i, :, :] = r_imp

    # Note that output of the flatten operation underlying formation of (2, 64) to 128 involves stacking 64-64, so
    # importance weighting should be applied im summation across the gap.
    importance_tally[0, :, :, :] *= np.reshape(rnn_in_weights_l[:num_final_filters], (int(output_size/2), 1, 1))
    importance_tally[0, :, :, :] *= np.reshape(rnn_in_weights_l[num_final_filters:], (int(output_size/2), 1, 1))

    importance_tally[1, :, :, :] *= np.reshape(rnn_in_weights_r[:num_final_filters], (int(output_size/2), 1, 1))
    importance_tally[1, :, :, :] *= np.reshape(rnn_in_weights_r[num_final_filters:], (int(output_size/2), 1, 1))

    importance_tally = np.sum(importance_tally, axis=1)

    filter_importance = np.zeros(importance_tally.shape)
    total_input_kernels = importance_tally.shape[0] * importance_tally.shape[2] * importance_tally.shape[2]

    for i in range(total_input_kernels):
        strongest = np.unravel_index(np.argmax(importance_tally), importance_tally.shape)
        importance_tally[strongest] = 0
        filter_importance[strongest] = i

    return filter_importance


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k_l, b_l = get_conv_weights_and_biases(params, left=True)
    k_r, b_r = get_conv_weights_and_biases(params, left=False)
    for i in range(1):
        # filter_order = get_most_consequential_conv_filters_for_conv(3, i, 0, k_l, b_l)
        # ordered_filters = reorder_filters(k_l[0], filter_order)
        filter_order = get_most_consequential_conv_filters_for_rnn(0, i, k_l, k_r, b_l, b_r,
                                                                   params["main_rnn_in/kernel:0"])
        ordered_filters_l = reorder_filters(k_l[0], filter_order[0])
        ordered_filters_r = reorder_filters(k_r[0], filter_order[0])
        display_cnn_filters([ordered_filters_l], first_is_coloured=True, mask_background=True)
        display_cnn_filters([ordered_filters_r], first_is_coloured=True, mask_background=True)


