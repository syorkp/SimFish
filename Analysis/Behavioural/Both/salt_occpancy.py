import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_salt_data import get_salt_data
from Analysis.Behavioural.Tools.anchored_scale_bar import AnchoredHScaleBar


def display_2d_kdf_salt_fish_position(fish_positions, salt_locations):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    salt_locations_flattened = np.concatenate(salt_locations, axis=0)

    relative_positions = salt_locations_flattened - fish_positions_flattened

    x = np.array([i[0] for i in list(relative_positions)])
    y = np.array([i[1] for i in list(relative_positions)])
    y = np.negative(y)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape))

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)
    plt.arrow(-300, 220, 0, 40, width=10, color="red")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Fish-Salt Relative Positions")
    plt.show()


def plot_salt_concentration_against_turn_away(fish_orientations, salt_locations, salt_concentrations):
    # TODO: Need to make it turn away.
    fish_turns = []
    for i in range(fish_orientations.shape[0]):
        fish_turns += list(fish_orientations[i, 1:] - fish_orientations[i, :-1])


    ...


fish_positions, fish_orientations, salt_source_locations, salt_concentrations = \
    get_salt_data("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 10)

# Salt-direction plot
plot_salt_concentration_against_turn_away(fish_positions, fish_orientations, salt_source_locations, salt_concentrations)

# Salt kdf.


