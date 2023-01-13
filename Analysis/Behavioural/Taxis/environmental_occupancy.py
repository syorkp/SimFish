import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data


def plot_environmental_occupancy_multiple_trials(model_name, assay_config, assay_id, n):

    fish_position_data = []

    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        fish_position_data.append(d["fish_position"])

    x = True

    # Plot as basic histogram
    # fish_position_data_flattened = np.concatenate(fish_position_data, axis=0)
    # plt.hist2d(fish_position_data_flattened[:, 0], fish_position_data_flattened[:, 1], bins=[np.arange(0, 3000, 5),
    #                                                                                          np.arange(0, 3000, 5)])
    # plt.show()

    # Plot as coloured line trajectories.
    for trace in fish_position_data:
        plt.scatter(trace[:, 0], trace[:, 1], alpha=0.1)
    plt.xlim(0, 3000)
    plt.ylim(0, 3000)
    plt.show()


if __name__ == "__main__":
    plot_environmental_occupancy_multiple_trials("dqn_scaffold_30_fixed_p-2", "Behavioural-Data-Free", "Naturalistic", 20)

    # plot_environmental_occupancy_multiple_trials("dqn_scaffold_30-2", "Behavioural-Data-Free", "Naturalistic", 20)

