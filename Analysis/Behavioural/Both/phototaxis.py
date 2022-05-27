import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import kde

from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files
from Analysis.Behavioural.Tools.get_repeated_data_parameter import get_parameter_across_trials
from Analysis.Behavioural.Tools.remove_near_wall_data import remove_near_wall_data_from_position_data


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
    k = kde.gaussian_kde([x, y])
    # yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    yi, xi = np.mgrid[0:env_variables["width"]:nbins * 1j, 0:env_variables["height"]:nbins * 1j]


    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape))

    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Environment Occupancy")
    plt.xlim(0, env_variables["width"])
    plt.ylim(0, env_variables["height"])

    # Display light-dark line
    dark_field_length = int(env_variables["width"] * env_variables["dark_light_ratio"])
    plt.hlines(dark_field_length, xmin=0, xmax=env_variables["width"])

    plt.show()


def plot_luminance_driven_choice(observations, actions, fish_positions, env_variables):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    if type(actions) is list:
        actions_flattened = np.concatenate(actions, axis=0)
    else:
        actions_flattened = actions
    if type(observations) is list:
        observations_flattened = np.concatenate(observations, axis=0)
    else:
        observations_flattened = observations

    # TODO: Decorrelate by removing wall sequences.
    directional_brightness = np.mean(observations_flattened, axis=1)
    directional_brightness = directional_brightness[:, 2, 0] - directional_brightness[:, 2, 1]

    going_left = (actions_flattened == 1) * 1 + (actions_flattened == 4) * 1
    going_left = (going_left > 0)
    brightness_left = directional_brightness[going_left]

    going_right = (actions_flattened == 2) * 1 + (actions_flattened == 5) * 1
    going_right = (going_right > 0)
    brightness_right = directional_brightness[going_right]

    mean_left = np.mean(brightness_left)
    mean_right = np.mean(brightness_right)

    plt.hist(brightness_left, bins=30, alpha=0.5, label="Turning Left", color="b")
    plt.axvline(mean_left, color="b", linestyle="dashed")
    plt.hist(brightness_right, bins=30, alpha=0.5, label="Turning Right", color="r")
    plt.axvline(mean_right, color="r", linestyle="dashed")
    plt.legend()
    plt.xlabel("Directional Brightness (Left is Positive)")
    plt.show()
    # TODO: Plot probability of turn as a function of directional brightness.
    # TODO: Make direction work automatically for both discrete and continuous.


def plot_oriention_against_directional_brightness(fish_orientations, observations):
    if type(fish_orientations) is list:
        fish_orientations_flattened = np.concatenate(fish_orientations, axis=0)
    else:
        fish_orientations_flattened = fish_orientations
    if type(observations) is list:
        observations_flattened = np.concatenate(observations, axis=0)
    else:
        observations_flattened = observations

    fish_turns = fish_orientations_flattened[1:] - fish_orientations_flattened[:1]

    # TODO: Decorrelate by removing wall sequences.
    directional_brightness = np.mean(observations_flattened, axis=1)
    directional_brightness = directional_brightness[:-1, 2, 0] - directional_brightness[:-1, 2, 1]
    fish_turns = np.array(fish_turns)

    # large_gradient = (((salt_gradients < -0.0001) + (salt_gradients > 0.0001)) * 1) > 0
    # salt_gradients = salt_gradients[large_gradient]
    # fish_turns = fish_turns[large_gradient]

    # to_move_upper = (fish_turns < 0) * (salt_gradients <= 0)
    # to_move_lower = (fish_turns < 0) * (salt_gradients > 0)
    # fish_turns *= (to_move_upper * -2) + 1
    # salt_gradients *= (to_move_upper * -2) + 1
    # fish_turns *= (to_move_lower * -2) + 1
    # salt_gradients *= (to_move_lower * -2) + 1

    plt.scatter(directional_brightness, fish_turns, alpha=0.01)
    plt.xlabel("Directional Brightness (Left is Positive)")
    plt.ylabel("Fish turn angle")
    # plt.xlim(0, 2)
    # plt.ylim(-0.00025, 0.00025)
    plt.show()
    x = True



compiled_fish_position_data = []
compiled_action_data = []
compiled_observation_data = []
compiled_orientation_data = []

for i in range(1, 5):
    # Display occupancy scatter plot and KDF.
    learning_params, env_variables, n, b, c = load_configuration_files(f"dqn_scaffold_14-{i}")
    fish_position_data = get_parameter_across_trials(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10, "fish_position")
    # plot_light_dark_occupancy(fish_position_data, env_variables)
    # plot_light_dark_occupancy_kdf(fish_position_data, env_variables)

    # Light gradient direction against turn laterality. NOTE: there are correlated factors such as presence of walls
    # and bkg_scatter dropoff towards edges
    action_data = get_parameter_across_trials(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10, "action")
    observation_data = get_parameter_across_trials(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10, "observation")
    plot_luminance_driven_choice(observation_data, action_data, fish_position_data, env_variables)
    reduced_fish_position, reduced_action_data, reduced_observation_data = remove_near_wall_data_from_position_data(fish_position_data,
                                                                                                                    env_variables["width"],
                                                                                                                    env_variables["height"],
                                                                                                                    300,
                                                                                                                    action_data,
                                                                                                                    observation_data)
    plot_luminance_driven_choice(reduced_observation_data, reduced_action_data, reduced_fish_position, env_variables)

    orientation_data = get_parameter_across_trials(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10, "fish_angle")
    plot_oriention_against_directional_brightness(orientation_data, observation_data)


    compiled_fish_position_data += fish_position_data
    compiled_action_data += action_data
    compiled_observation_data += observation_data
    compiled_orientation_data += orientation_data

# For all models
# plot_light_dark_occupancy(compiled_fish_position_data, env_variables)
# plot_light_dark_occupancy_kdf(compiled_fish_position_data, env_variables)
plot_luminance_driven_choice(compiled_observation_data, compiled_action_data, compiled_fish_position_data, env_variables)
plot_oriention_against_directional_brightness(compiled_orientation_data, compiled_observation_data)
