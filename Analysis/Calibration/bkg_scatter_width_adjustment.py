import numpy as np
import matplotlib.pyplot as plt


def compute_sum_of_bkground(width, bkg_scatter):

    xp, yp = np.arange(width), np.arange(width)

    i, j = xp[:, None], yp[None, :]
    decay_rate = 0.001

    x = 0
    y = 0
    positional_mask = (((x - i) ** 2 + (y - j) ** 2) ** 0.5)  # Measure of distance from fish at every point.
    desired_scatter = np.exp(-decay_rate * positional_mask)
    distance = np.linspace(0, width, width)

    scatter = desired_scatter[0, :]
    total_scatter = np.sum(scatter * bkg_scatter)
    # print(f"Total scatter: {total_scatter}")
    return total_scatter
    # plt.plot(distance, desired_scatter[0, :])
    # plt.show()


def build_bk_scatter_model(bkg_scatter):
    widths = np.linspace(100, 10000, 100).astype(int)
    widths2 = np.linspace(100, 10000, 100).astype(int)
    scatters = [compute_sum_of_bkground(width, bkg_scatter) for width in widths]

    model = np.polyfit(widths, scatters, 5)
    p = np.poly1d(model)
    print(model)
    p_widths = p(widths2)
    # [ 1.32283913e-18 -4.10522256e-14  4.92470049e-10 -2.86744090e-06
    #   8.22376164e-03  4.07923942e-01]

    # bkg_scatter_adjustment = (p(width)/p(1500)) * bkg_scatter

    plt.plot(widths, scatters)
    plt.plot(widths2, p_widths)
    plt.show()

    print(p(1500))
