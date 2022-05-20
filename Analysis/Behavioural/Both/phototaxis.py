import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files


def plot_light_dark_occupancy(fish_positions, env_variables):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)

    x = fish_positions_flattened[:, 0]
    y = fish_positions_flattened[:, 1]

    plt.hist2d(x, y, bins=[np.arange(0, env_variables["width"], 5), np.arange(0, env_variables["height"], 5)])
    plt.show()


learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_14-1")

