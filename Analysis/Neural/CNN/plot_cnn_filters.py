import copy

import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases


def normalise_layers(layers):
    max_val = max([np.max(layer) for layer in layers])
    min_val = min([np.min(layer) for layer in layers])

    to_scale = max(abs(max_val), abs(min_val))

    layers = [layer/to_scale for layer in layers]
    return layers


def normalise_layers2(layers):
    max_val = max([np.max(layer) for layer in layers])
    min_val = min([np.min(layer) for layer in layers])

    if min_val < 0:
        layers = [layer + abs(min_val) for layer in layers]

    to_scale = abs(max_val) + abs(min_val)

    layers = [layer/to_scale for layer in layers]
    return layers


def order_cnn_filters_by_kernel_location(layer, colour_channel):
    """Given array of (kernel_size . n_channels . n_units), rearranges along n_units dimension according to where
    centre of mass of units are.

    INITIAL ATTEMPT AT THIS FAILED - NOT SURE IS A GOOD WAY IN WHICH TO ORGANISE THEM ANYWAY - TEND TO HAVE MULTIPLE
    CENTRES OF MASS.
    """

    just_selected_channel = layer[:, :, colour_channel]
    just_selected_channel_positive = just_selected_channel * (just_selected_channel > 0)
    points = np.arange(0, layer.shape[0])
    points = np.expand_dims(points, 1)
    kernel_weighting = just_selected_channel_positive * points
    kernel_centres = np.sum(kernel_weighting, axis=0)/np.sum(just_selected_channel_positive, axis=0)
    x = True


def display_cnn_filters(layers, first_is_coloured, mask_background, show_negative_alongside=True, order_filters=False,
                        mask_red=False, normalisation_mode="upscale"):
    """Displays kernels of CNN filters."""
    # Normalise layers
    if normalisation_mode == "rescale":
        layers = normalise_layers(layers)
    elif normalisation_mode == "upscale":
        layers = normalise_layers2(layers)
    else:
        pass

    for i, layer in enumerate(layers):
        n_units = layer.shape[-1]
        layer = np.swapaxes(layer, 0, 1)

        if i == 0 and first_is_coloured:
            layer = np.swapaxes(layer, 0, 1)
            layer = np.swapaxes(layer, 1, 2)
            layer = np.swapaxes(layer, 0, 1)
            # Make RGB.
            layer = np.concatenate((layer[:, :, 0:1], layer[:, :, 2:3], layer[:, :, 1:2]), axis=2)
            if mask_background:
                layer[:, :, 1] = 0
            if mask_red:
                layer[:, :, 0] = 0


            if order_filters:
                layer = order_cnn_filters_by_kernel_location(layer, 0)

            layer_positive = layer * (layer >= 0)
            layer_negative = layer * (layer <= 0)

            if show_negative_alongside and normalisation_mode != "upscale":
                compiled_layers = []
                for i in range(len(layer_positive)):
                    compiled_layers.append(layer_positive[i])
                    compiled_layers.append(np.absolute(layer_negative[i]))
                compiled_layers = np.array(compiled_layers)
                plt.title("Positive and Negative Filters")
                plt.imshow(compiled_layers)
                plt.xlabel("Point Across kernel")
                plt.ylabel("Unit")
                plt.tight_layout()
                plt.show()

            else:
                plt.title("Positive Filters")
                plt.imshow(layer_positive)
                plt.xlabel("Point Across kernel")
                plt.ylabel("Unit")
                plt.tight_layout()
                plt.show()

                plt.title("Negative Filters")
                plt.imshow(np.absolute(layer_negative))
                plt.xlabel("Point Across kernel")
                plt.ylabel("Unit")
                plt.tight_layout()
                plt.show()
        else:
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
    display_cnn_filters(k, first_is_coloured=True, mask_background=True)
