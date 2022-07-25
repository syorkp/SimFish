import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from Analysis.load_data import load_data


def plot_pca_trajectory(activity_data, timepoints_to_label=None):
    # step_data = activity_data[:, t:t+1]
    pca = PCA(n_components=2)
    pca.fit(activity_data)
    pca_components = pca.components_
    plt.scatter(pca_components[0], pca_components[1], c=np.arange(len(pca_components[0])))
    if timepoints_to_label is not None:
        pca_points_at_timestamps = np.array([pca_components[:, i] for i in range(pca_components.shape[1]) if i in timepoints_to_label])
        plt.scatter(pca_points_at_timestamps[:, 0], pca_points_at_timestamps[:, 1], marker="x", color="r")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    rnn_data_full = np.array([])
    for i in range(1, 11):
        data = load_data("dqn_scaffold_18-2", "Behavioural-Data-Free", f"Naturalistic-{i}")
        # data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
        rnn_data = data["rnn_state_actor"][:, 0, 0, :]
        rnn_data = np.swapaxes(rnn_data, 0, 1)
        rnn_data_full = np.concatenate((rnn_data_full, rnn_data))
        consumption_points = [i for i in range(len(data["consumed"])) if data["consumed"][i]]
        interruptions = [i for i in range(len(data["consumed"])) if i > 100 and i % 10 == 0]
    plot_pca_trajectory(rnn_data, timepoints_to_label=consumption_points)



