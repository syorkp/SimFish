"""
To determine how to adjust scatter mask to incorporate implicit scatter (which is a result of the way the visual system
works.
"""

import matplotlib.pyplot as plt
import numpy as np


def get_final_scatter_mask2(i, j, x, y, light_decay_rate, theta):
    positional_mask = (((x - i) ** 2 + (y - j) ** 2) ** 0.5)
    desired_scatter = np.exp(-light_decay_rate * positional_mask)
    implicit_scatter = np.sin(theta) * positional_mask
    implicit_scatter[implicit_scatter < 1] = 1
    adjusted_scatter = desired_scatter * implicit_scatter
    return adjusted_scatter


def get_final_scatter_mask(i, j, x, y, light_decay_rate, theta):
    desired_scatter = np.exp(-light_decay_rate * (((x - i) ** 2 + (y - j) ** 2) ** 0.5))
    implicit_scatter = np.sin(theta) * (((x - i) ** 2 + (y - j) ** 2) ** 0.5)
    implicit_scatter[implicit_scatter < 1] = 1
    adjusted_scatter = desired_scatter * implicit_scatter
    return adjusted_scatter

def scatter(i, j, x, y, light_decay_rate):
    return np.exp(-light_decay_rate * (((x - i) ** 2 + (y - j) ** 2) ** 0.5))


def implicit_decay(i, j, x, y, theta):
    s = np.sin(theta) * (((x - i) ** 2 + (y - j) ** 2) ** 0.5)
    s[s < 1] = 1
    s = 1/s
    return s

def show_mask(mask):
    extent = [0, 1000, 0, 1000]

    plt.clf()
    plt.imshow(mask, extent=extent, origin='lower')
    plt.show()


xp, yp = np.arange(1000), np.arange(1000)

theta_vals = np.linspace(0.001, 0.02, 1000)
theta_vals = np.expand_dims(theta_vals, 1)
theta_vals = np.repeat(theta_vals, 1000, 1)
d_vals = np.linspace(0, 1500, 1000)
d_vals = np.expand_dims(d_vals, 0)
d_vals = np.repeat(d_vals, 1000, 0)

real_scatter = scatter(xp[:, None], yp[None, :], 500, 50, 0.01)

s = np.sin(theta_vals) * d_vals

s[s < 1] = 1

s = 1/s

implicit_d = implicit_decay(xp[:, None], yp[None, :], 500, 50, 0.01)

final = get_final_scatter_mask2(xp[:, None], yp[None, :], 500, 50, 0.01, 0.01)

# show_mask(s)
# show_mask(real_scatter)
# show_mask(implicit_d)
show_mask(final)


x = True



