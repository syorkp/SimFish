import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap

from Analysis.load_data import load_data
from Analysis.Neural.Tools.remove_inconsequential_neurons import remove_those_with_no_output, remove_those_with_no_output_advantage_only


def plot_xxx_trajectory_multiple_trials(activity_data, algorithm, timepoints_to_label=None):
    flattened_activity_data = np.concatenate((activity_data), axis=1)
    xxx = algorithm(n_components=2)
    xxx.fit(np.swapaxes(flattened_activity_data, 0, 1))
    xxx_components = xxx.kernel_pca_.alphas_[:, :]

    split_colours = np.array([])
    for i in range(len(activity_data)):
        split_colours = np.concatenate((split_colours, np.arange(len(activity_data[i][0]))))

    plt.scatter(xxx_components[:, 0], xxx_components[:, 1], c=split_colours)

    timepoints_to_label = None
    if timepoints_to_label is not None:
        # Adjust timepoints to label so follows indexing of RNN data
        len_of_each = [0] + [len(c[0]) for c in activity_data][:-1]
        flattened_timepoints_to_label = []
        for i, c in enumerate(timepoints_to_label):
            adjustment = np.sum(len_of_each[:i+1])
            flattened_timepoints_to_label += [p + adjustment for p in timepoints_to_label[i]]
        pca_points_at_timestamps = np.array([xxx_components[:, i] for i in range(xxx_components.shape[1]) if i in flattened_timepoints_to_label])
        plt.scatter(pca_points_at_timestamps[:, 0], pca_points_at_timestamps[:, 1], marker="x", color="r")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    rnn_data_full = []
    consumption_points = []
    for i in range(1, 2):
        data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Endless", f"Naturalistic-{i}")
        # data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
        rnn_data = data["rnn_state_actor"][:, 0, 0, :]
        rnn_data = np.swapaxes(rnn_data, 0, 1)
        consumption_points.append([i for i in range(len(data["consumed"])) if data["consumed"][i]])
        rnn_data_full.append(rnn_data)

    # plot_pca_trajectory(rnn_data_full, timepoints_to_label=consumption_points)
    plot_xxx_trajectory_multiple_trials(rnn_data_full, Isomap, consumption_points)
    # data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Endless", f"Naturalistic-1")
    # rnn_data = np.swapaxes(data["rnn_state_actor"][:, 0, 0, :], 0, 1)
    # # reduced_rnn_data = remove_those_with_no_output(rnn_data, "dqn_scaffold_18-2", "dqn_18_2", proportion_to_remove=0.2)
    # reduced_rnn_data = rnn_data[:256]
    # # reduced_rnn_data = rnn_data[256:]
    # # reduced_rnn_data = rnn_data
    #
    # plot_pca_trajectory(reduced_rnn_data, timepoints_to_label=None)

