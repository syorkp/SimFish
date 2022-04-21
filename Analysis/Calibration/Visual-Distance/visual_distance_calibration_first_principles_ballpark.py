"""
Doesn't include effect of shot noise so gives no measure of uncertainty. Just mean values for each.
For finding ballpark values... Gets the max scatter signal alongside max prey signal (with scatter included). VALIDATED for bkg_scatter=0
"""

import numpy as np
import matplotlib.pyplot as plt


def prey_signal(L, d, decay_constant=0.0006):
    return L * np.exp(-decay_constant * d)


def prey_signal_all_ds(L, distance, decay_constant=0.0006):
    d_range = np.linspace(0, distance, 1000)
    prey_signals = [prey_signal(L, d, decay_constant) for d in d_range]
    return prey_signals


def scatter_signal_all(max_d, rf_size, bkg_scatter, decay_constant):
    d_range = np.linspace(1, max_d, int(max_d-1))

    point_width = 2 * d_range * np.tan(rf_size / 2)
    distance_scaling = np.exp(-decay_constant * d_range) * bkg_scatter
    point_width = np.clip(point_width, 1, 10000)
    point_width += (point_width > 1) * 2
    # point_width = np.floor(point_width).astype(int)
    photons = np.sum(distance_scaling * point_width)

    return photons


decay = 0.01
max_distance_s = (1500**2 + 1500**2) ** 0.5
luminance = 200
distance = 600
bkg_scatter = 0.1
rf_size = 0.0133

plt.hlines(scatter_signal_all(max_distance_s, rf_size, bkg_scatter, decay), 0, distance)
plt.plot(np.linspace(0, distance, 1000), prey_signal_all_ds(luminance, distance, decay) + scatter_signal_all(max_distance_s, rf_size, bkg_scatter, decay))
plt.show()







