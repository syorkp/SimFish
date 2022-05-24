import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import kde

from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files
from Analysis.Behavioural.Tools.get_repeated_data_parameter import get_parameter_across_trials


def plot_light_dark_occupancy(fish_positions, env_variables):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)

    x = fish_positions_flattened[:, 0]
    y = fish_positions_flattened[:, 1]

    plt.hist2d(x, y, bins=[np.arange(0, env_variables["width"], 5), np.arange(0, env_variables["height"], 5)])

    # Display light-dark line
    dark_field_length = int(env_variables["width"] * env_variables["dark_light_ratio"])
    plt.hlines(dark_field_length, xmin=0, xmax=env_variables["width"])

    plt.show()


def plot_light_dark_occupancy_kdf(fish_positions, env_variables):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)

    x = fish_positions_flattened[:, 0]
    y = fish_positions_flattened[:, 1]

    # y = np.negative(y)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape))

    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Fish-Salt Relative Positions")

    # Display light-dark line
    dark_field_length = int(env_variables["width"] * env_variables["dark_light_ratio"])
    plt.vlines(dark_field_length, ymin=0, ymax=env_variables["width"])

    plt.show()


def plot_luminance_driven_choice(observations, actions, fish_positions, env_variables):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    actions_flattened = np.concatenate(actions, axis=0)
    observations_flattened = np.concatenate(observations, axis=0)

    # TODO: Decorrelate by removing wall sequences.
    directional_brightness = np.mean(observations_flattened, axis=1)
    directional_brightness = directional_brightness[:, 2, 0] - directional_brightness[:, 2, 1]

    going_left = (actions_flattened == 1) * 1 + (actions_flattened == 4) * 1
    going_left = (going_left > 0)
    brightness_left = directional_brightness[going_left]

    going_right = (actions_flattened == 2) * 1 + (actions_flattened == 5) * 1
    going_right = (going_right > 0)
    brightness_right = directional_brightness[going_right]

    plt.hist(brightness_left, bins=20)
    plt.show()

    plt.hist(brightness_right, bins=20)
    plt.show()
    # TODO: Plot probability of turn as a function of directional brightness.
    # TODO: Make direction work automatically for both discrete and continuous.
    x = True


with open("../../../Configurations/Training-Configs/dqn_scaffold_14/36_env.json", 'r') as f:
    env_variables = json.load(f)
# learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_14-1")
fish_position_data = get_parameter_across_trials("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 10, "fish_position")
plot_light_dark_occupancy(fish_position_data, env_variables)
plot_light_dark_occupancy_kdf(fish_position_data, env_variables)

# Light gradient direction against turn laterality. NOTE: there are correlated factors such as presence of walls
# and bkg_scatter dropoff towards edges
action_data = get_parameter_across_trials("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 10, "action")
observation_data = get_parameter_across_trials("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 10, "observation")
plot_luminance_driven_choice(observation_data, action_data, fish_position_data, env_variables)