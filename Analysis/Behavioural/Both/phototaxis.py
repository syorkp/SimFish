import numpy as np
import matplotlib.pyplot as plt

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


def plot_luminance_driven_choice(observations, actions, fish_positions, env_variables):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    actions_flattened = np.concatenate(actions, axis=0)
    observations_flattened = np.concatenate(observations, axis=0)

    # TODO: Decorrelate by removing wall sequences.
    directional_brightness = np.mean(observations_flattened, axis=1)
    directional_brightness = directional_brightness[:, 2, 0] - directional_brightness[:, 2, 1]

    # TODO: Plot probability of turn as a function of directional brightness.
    # TODO: Make direction work automatically for both discrete and continuous.
    x = True


learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_14-1")
fish_position_data = get_parameter_across_trials("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 3, "fish_position")
plot_light_dark_occupancy(fish_position_data, env_variables)

action_data = get_parameter_across_trials("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 3, "action")
observation_data = get_parameter_across_trials("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 3, "observation")
plot_luminance_driven_choice(observation_data, action_data, fish_position_data, env_variables)