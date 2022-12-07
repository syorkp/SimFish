import numpy as np


def reorder_filters(filters, order):
    """Takes ndarray of filters (kernel_size, channels, n_units) and order (channels, n_units).
    Orders along the n_units axis.
    """
    collapsed_order = np.sum(order, axis=0)
    ordered_filters = []

    for c in range(len(collapsed_order)):
        first = np.argmin(collapsed_order)
        ordered_filters.append(filters[:, :, first])
        collapsed_order[first] = 100000

    ordered_filters = np.array(ordered_filters)
    ordered_filters = np.swapaxes(ordered_filters, 1, 2)
    return ordered_filters


