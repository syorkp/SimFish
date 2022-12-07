import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage

from Analysis.Behavioural.Tools.get_repeated_data_parameter import get_parameter_across_trials


def show_mean_stimulus_level_across_arena(observations, fish_positions):
    sigma_y = 10.0
    sigma_x = 10.0
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    observations_flattened = np.concatenate(observations, axis=0)

    mean_red_stimulation = np.mean(observations_flattened[:, :, 0, :], axis=(1, 2))

    fish_positions_flattened = fish_positions_flattened.astype(int)

    z = np.ones((1500, 1500))
    z[fish_positions_flattened[:, 0], fish_positions_flattened[:, 1]] = mean_red_stimulation
    plt.imshow(z)
    plt.show()

    sigma = [sigma_y, sigma_x]
    y = sp.ndimage.filters.gaussian_filter(z, sigma, mode='constant')

    plt.imshow(y)
    plt.show()

fish_position_tally = []
observation_tally = []
for i in range(1, 5):
    fish_position_data = get_parameter_across_trials(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10, "fish_position")
    observation_data = get_parameter_across_trials(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10, "observation")

    fish_position_tally += fish_position_data
    observation_tally += observation_data

show_mean_stimulus_level_across_arena(observation_tally, fish_position_tally)


