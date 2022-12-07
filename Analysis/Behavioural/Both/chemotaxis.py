import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_salt_data import get_salt_data
from Analysis.Behavioural.Tools.anchored_scale_bar import AnchoredHScaleBar
from Analysis.Behavioural.Tools.random_data_generators import generate_random_fish_position_data


def display_2d_kdf_salt_fish_position(fish_positions, salt_locations):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    salt_locations_flattened = np.concatenate(salt_locations, axis=0)

    relative_positions = salt_locations_flattened - fish_positions_flattened

    x = np.array([i[0] for i in list(relative_positions)])
    y = np.array([i[1] for i in list(relative_positions)])
    # y = np.negative(y)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape))

    plt.scatter(0, 0, color="red")
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Fish-Salt Relative Positions")
    plt.show()


def plot_fish_salt_distance_density(fish_positions, salt_locations, w=1500, h=1500):
    d_max = (w ** 2 + h ** 2) ** 0.5
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    salt_locations_flattened = np.concatenate(salt_locations, axis=0)

    relative_positions = salt_locations_flattened - fish_positions_flattened
    distances = (relative_positions[:, 0] ** 2 + relative_positions[:, 1] ** 2) ** 0.5

    random_fish_positions = generate_random_fish_position_data(n=fish_positions_flattened.shape[0], w=w, h=h)
    relative_positions = salt_locations_flattened - random_fish_positions
    distances_simulated = (relative_positions[:, 0] ** 2 + relative_positions[:, 1] ** 2) ** 0.5

    plt.title("Histogram of Simulated Fish Position-Salt Source Distance")
    plt.hist(distances, bins=100, range=[0, d_max], alpha=0.5, color="b", label="Fish Distances")
    plt.hist(distances_simulated, bins=100, range=[0, d_max], alpha=0.5, color="r", label="Simulated Fish Distances")
    plt.xlabel("Distance From Salt Source")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def plot_salt_concentration_against_turn_laterality(fish_orientations, fish_positions, salt_locations, salt_concentrations):
    fish_turns = []
    salt_gradients = []
    for i in range(len(fish_orientations)):
        fish_turns += list(fish_orientations[i][1:] - fish_orientations[i][:-1])
        salt_gradients += list(salt_concentrations[i][1:] - salt_concentrations[i][:-1])

    fish_turns = np.array(fish_turns)
    salt_gradients = np.array(salt_gradients)

    large_gradient = (((salt_gradients < -0.0001) + (salt_gradients > 0.0001)) * 1) > 0
    salt_gradients = salt_gradients[large_gradient]
    fish_turns = fish_turns[large_gradient]

    to_move_upper = (fish_turns < 0) * (salt_gradients <= 0)
    to_move_lower = (fish_turns < 0) * (salt_gradients > 0)
    fish_turns *= (to_move_upper * -2) + 1
    salt_gradients *= (to_move_upper * -2) + 1
    fish_turns *= (to_move_lower * -2) + 1
    salt_gradients *= (to_move_lower * -2) + 1

    plt.scatter(fish_turns, salt_gradients)
    plt.xlim(0, 2)
    plt.ylim(-0.00025, 0.00025)
    plt.show()


def plot_salt_concentration_against_turn_direction(fish_orientations, fish_positions, salt_locations, salt_concentrations):
    fish_turns = []
    salt_gradients = []
    for i in range(len(fish_orientations)):
        fish_turns += list(fish_orientations[i][1:] - fish_orientations[i][:-1])
        salt_gradients += list(salt_concentrations[i][1:] - salt_concentrations[i][:-1])

    fish_turns = np.array(fish_turns)
    salt_gradients = np.array(salt_gradients)

    # large_gradient = (((salt_gradients < -0.0001) + (salt_gradients > 0.0001)) * 1) > 0
    # salt_gradients = salt_gradients[large_gradient]
    # fish_turns = fish_turns[large_gradient]

    # to_move_upper = (fish_turns < 0) * (salt_gradients <= 0)
    # to_move_lower = (fish_turns < 0) * (salt_gradients > 0)
    # fish_turns *= (to_move_upper * -2) + 1
    # salt_gradients *= (to_move_upper * -2) + 1
    # fish_turns *= (to_move_lower * -2) + 1
    # salt_gradients *= (to_move_lower * -2) + 1

    plt.scatter(fish_turns, salt_gradients, alpha=0.1)
    # plt.xlim(0, 2)
    # plt.ylim(-0.00025, 0.00025)
    plt.show()


def plot_salt_driven_choice(salt_concentrations, actions, fish_positions, env_variables):
    # TODO: NOT FINSIHED OR USED.
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    if type(actions) is list:
        actions_flattened = np.concatenate(actions, axis=0)
    else:
        actions_flattened = actions
    if type(salt_concentrations) is list:
        salt_concentrations_flattened = np.concatenate(salt_concentrations, axis=0)
    else:
        salt_concentrations_flattened = salt_concentrations

    # TODO: Decorrelate by removing wall sequences.
    salt_gradient = np.mean(salt_concentrations_flattened, axis=1)
    salt_gradient = salt_gradient[1:] - salt_gradient[:-1]

    going_left = (actions_flattened == 1) * 1 + (actions_flattened == 4) * 1
    going_left = (going_left > 0)
    salt_left = salt_gradient[going_left]

    going_right = (actions_flattened == 2) * 1 + (actions_flattened == 5) * 1
    going_right = (going_right > 0)
    salt_right = salt_gradient[going_right]

    mean_left = np.mean(salt_left)
    mean_right = np.mean(salt_right)

    plt.hist(salt_left, bins=30, alpha=0.5, label="Turning Left", color="b")
    plt.axvline(mean_left, color="b", linestyle="dashed")
    plt.hist(salt_right, bins=30, alpha=0.5, label="Turning Right", color="r")
    plt.axvline(mean_right, color="r", linestyle="dashed")
    plt.legend()
    plt.xlabel("Directional Brightness (Left is Positive)")
    plt.show()


compiled_fish_positions = []
compiled_fish_orientations = []
compiled_salt_source_locations = []
compiled_salt_concentrations = []
for i in range(1, 5):
    fish_positions, fish_orientations, salt_source_locations, salt_concentrations = \
        get_salt_data(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10)
    # display_2d_kdf_salt_fish_position(fish_positions, salt_source_locations)
    # plot_fish_salt_distance_density(fish_positions, salt_source_locations)
    plot_salt_concentration_against_turn_direction(fish_orientations, fish_positions,
                                                   salt_source_locations, salt_concentrations)
    compiled_fish_positions += fish_positions
    compiled_fish_orientations += fish_orientations
    compiled_salt_source_locations += salt_source_locations
    compiled_salt_concentrations += salt_concentrations

# Salt kdf.
# display_2d_kdf_salt_fish_position(compiled_fish_positions, compiled_salt_source_locations)
# plot_fish_salt_distance_density(compiled_fish_positions, compiled_salt_source_locations)

# Salt-direction plot
plot_salt_concentration_against_turn_direction(compiled_fish_orientations, compiled_fish_positions,
                                                compiled_salt_source_locations, compiled_salt_concentrations)
# plot_salt_concentration_against_turn_laterality(compiled_fish_orientations, compiled_fish_positions,
#                                                 compiled_salt_source_locations, compiled_salt_concentrations)


