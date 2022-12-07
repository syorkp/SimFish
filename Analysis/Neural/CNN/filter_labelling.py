import numpy as np

from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.Connectivity.get_conv_weights import get_conv_weights_and_biases
from Analysis.Neural.CNN.plot_cnn_filters import display_cnn_filters


"""
Labels: 
  - contiguous uv - of various sizes
  - contiguous red - of various sizes
  - gradient of UV
  - gradient of red
  - UV surrounded by nothing
  - UV surrounded by red.

"""


def label_filters_contiguous(kernel, length, colour_index, colour_threshold):
    """Kernel received has shape (kernel_length, channels)"""

    contiguous_points = []

    required_kernel = np.ones((length)) * colour_threshold
    for i in range(0, (kernel.shape[0]+1)-length):
        tested_kernel_point = kernel[i:i+length, colour_index]
        above_requirement = (tested_kernel_point > required_kernel)
        if np.all(above_requirement):
            contiguous_points.append([i for i in range(i, i+length)])

    return contiguous_points


def label_filters_surrounding(kernel, length, surrounded_points, colour_index, colour_threshold):
    # Here, colour threshold should be used to differentiate points that are surrounded by nothing, and those which have
    # a negative threshold surrounding them.
    ...


def label_filters_gradient(kernel, length, colour_index):
    # DO in both directions
    gradient_left = kernel[:-1, colour_index] - kernel[1:, colour_index]
    gradient_right = kernel[1:, colour_index] - kernel[:-1, colour_index]

    negative_gradient_locations = []
    positive_gradient_locations = []

    streak_l = 0
    streak_r = 0
    for i in range(0, (gradient_left.shape[0]+1)-length):
        if i == 0:
            pass
        else:
            if gradient_left[i] < previous_val_l:
                streak_l += 1
            if gradient_right[i] < previous_val_r:
                streak_r += 1

            if streak_l == length:
                negative_gradient_locations.append([j for j in range(i-length, i)])
                streak_l = 0
            if streak_r == length:
                positive_gradient_locations.append([j for j in range(i-length, i)])
                streak_r = 0

        previous_val_l = gradient_left[i]
        previous_val_r = gradient_right[i]

    return positive_gradient_locations, negative_gradient_locations


def get_all_filter_labels(kernel, positive_colour_threshold, negative_colour_thresholds):
    all_labels = []

    # Label Contiguous colours
    for colour_i in range(0, 3):
        colour = ["Red", "UV", "Red2"][colour_i]

        for tested_length in range(2, kernel.shape[0]):
            locations = label_filters_contiguous(kernel, tested_length, colour_index=colour_i,
                                                 colour_threshold=positive_colour_threshold[colour_i])
            for location in locations:
                all_labels.append(f"Contiguous {colour}: Length {tested_length}, Locations: {location}")
                # TODO: Look for surroundings.

            positive_gradient_locations, negative_gradient_locations = label_filters_gradient(kernel, tested_length,
                                                                                              colour_index=colour_i)
            for location in positive_gradient_locations:
                all_labels.append(f"Contiguous {colour}: Length {tested_length}, Locations: {location}")

    return all_labels


def get_thresholds(kernels, percentile=90):
    positive_thresholds = np.zeros((3))
    negative_thresholds = np.zeros((3))

    for c in range(3):
        positive_values = np.sort(kernels[:, c][(kernels[:, c] > 0)])
        negative_values = np.sort(kernels[:, c][(kernels[:, c] <= 0)])

        # TODO: Find threshold
        num_positive = len(positive_values)
        threshold_positive = num_positive * percentile/100
        num_negative = len(negative_values)
        threshold_negative = num_negative * percentile/100

        for i, val in enumerate(positive_values):
            if i > threshold_positive:
                positive_thresholds[c] = val
                break

        for i, val in enumerate(reversed(negative_values)):
            if i > threshold_negative:
                negative_thresholds[c] = val
                break

    return positive_thresholds, negative_thresholds


def label_all_filters(kernels):
    # TODO: determine colour_threshold  - should be based on percentiles.
    positive_thresholds, negative_thresholds = get_thresholds(kernels)

    compiled_labels = []
    for unit_number in range(kernels.shape[-1]):
        labels = get_all_filter_labels(kernels[:, :, unit_number], positive_thresholds, negative_thresholds)
        compiled_labels.append(labels)
    return compiled_labels


def order_filters_by_label(kernels, labels, length=5):
    """Just orders by location of first filter of size 3"""
    location_of_first = np.zeros((16))
    for i, label in enumerate(labels):
        size_3 = [l for l in label if f"Length {length}" in l and "UV" in l]
        if len(size_3) > 0:
            first_size_3 = size_3[0]
            location = first_size_3.split("Locations: ")[-1][1:-1]
            init_loc = int(location.split(",")[0]) + 1
            location_of_first[i] = init_loc

    new_kernel_array = np.zeros((16, 3, 16))
    counter = 0
    for m in range(int(max(location_of_first)+1)):
        num = np.sum((location_of_first == m) * 1)
        if num > 0:
            selected_kernels = kernels[:, :, location_of_first == m]
            counter += num
            new_kernel_array[:, :, counter-num:counter] = selected_kernels

    return new_kernel_array


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k, b = get_conv_weights_and_biases(params, left=True)
    compiled_labels = label_all_filters(k[0])
    ordered_k0 = order_filters_by_label(k[0], compiled_labels, length=4)
    display_cnn_filters([k[0]], first_is_coloured=True, mask_background=True, normalisation_mode="rescale", mask_red=True)
