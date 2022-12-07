import matplotlib.pyplot as plt
import numpy as np

from Analysis.Connectivity.load_network_variables import load_network_variables


# Thought - filter size is limiting recognition for fish.

def visualise_filters(data, filter_num, side="l"):
    filters = data[f"main_conv{filter_num}{side}/kernel:0"]
    bias = data[f"main_conv{filter_num}{side}/bias:0"]
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    for i in range(filters.shape[2]):
        f = filters[:, :, i]
        for j in range(3):
            ax = plt.subplot()
            # ax.set_xticks([])
            # ax.set_yticks([])
            plt.imshow(np.expand_dims([fil[j] for fil in f], axis=0), aspect="auto", cmap="gray")
            plt.show()


v = load_network_variables("even_prey_ref-5", "1")
visualise_filters(v, 3)
x = True