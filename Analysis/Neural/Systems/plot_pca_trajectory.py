import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from Analysis.load_data import load_data
from Analysis.Neural.Tools.remove_inconsequential_neurons import remove_those_with_no_output, \
    remove_those_with_no_output_advantage_only
from Analysis.Behavioural.Tools.label_behavioural_context import label_behavioural_context_multiple_trials, \
    get_behavioural_context_name_by_index

from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces


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


def estimate_gaussian(dataset):

    mu = np.mean(dataset) # moyenne cf mu
    sigma = np.std(dataset) # écart_type/standard deviation
    limit = sigma * 10

    min_threshold = mu - limit
    max_threshold = mu + limit

    return mu, sigma, min_threshold, max_threshold


def plot_pca_trajectory_multiple_trials(activity_data, timepoints_to_label=None, display_numbers=True,
                                        context_name="No Label", self_normalise_activity_data=True, n_components=2,
                                        exclude_outliers=True):
    flattened_activity_data = np.concatenate((activity_data), axis=1)
    if self_normalise_activity_data:
        flattened_activity_data = normalise_within_neuron_multiple_traces(flattened_activity_data)

    pca = PCA(n_components=n_components)
    pca.fit(flattened_activity_data)
    pca_components = pca.components_[:, :]

    split_colours = np.array([])
    for i in range(len(activity_data)):
        split_colours = np.concatenate((split_colours, np.arange(len(activity_data[i][0]))))

    # Phase space
    fig, ax = plt.subplots(figsize=(10, 10))
    if display_numbers:
        for i in range(len(pca_components[0])):
            ax.annotate(i, (pca_components[0, i], pca_components[1, i]))

    if n_components == 2:
        plt.scatter(pca_components[0], pca_components[1], c=split_colours)
    elif n_components == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pca_components[0], pca_components[1], pca_components[2], c=split_colours)
    else:
        print("Unsupported number of components")

    if timepoints_to_label is not None:
        # Adjust timepoints to label so follows indexing of RNN data
        len_of_each = [0] + [len(c[0]) for c in activity_data][:-1]
        flattened_timepoints_to_label = []
        for i, c in enumerate(timepoints_to_label):
            adjustment = np.sum(len_of_each[:i+1])
            flattened_timepoints_to_label += [p + adjustment for p in timepoints_to_label[i]]
        pca_points_at_timestamps = np.array([pca_components[:, i] for i in range(pca_components.shape[1]) if i in
                                             flattened_timepoints_to_label])
        try:
            if n_components == 2:
                plt.scatter(pca_points_at_timestamps[:, 0], pca_points_at_timestamps[:, 1], marker="x", color="r")
            elif n_components == 3:
                ax.scatter(pca_points_at_timestamps[:, 0], pca_points_at_timestamps[:, 1],
                           pca_points_at_timestamps[:, 2], marker="x", color="r")
        except IndexError:
            pass

    if n_components == 2:
        plt.colorbar()

    plt.title("PCA Phase Space: " + context_name)
    plt.show()


    # Trajectory space
    pca_components = pca_components[:, 1:] - pca_components[:, :-1]
    fig, ax = plt.subplots(figsize=(10, 10))
    if display_numbers:
        for i in range(len(pca_components[0])):
            ax.annotate(i, (pca_components[0, i], pca_components[1, i]))

    if n_components == 2:
        plt.scatter(pca_components[0], pca_components[1], c=split_colours[1:])
    elif n_components == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pca_components[0], pca_components[1], pca_components[2], c=split_colours[1:])
    else:
        print("Unsupported number of components")

    if timepoints_to_label is not None:
        # Adjust timepoints to label so follows indexing of RNN data
        len_of_each = [0] + [len(c[0]) for c in activity_data][:-1]
        flattened_timepoints_to_label = []
        for i, c in enumerate(timepoints_to_label):
            adjustment = np.sum(len_of_each[:i+1])
            flattened_timepoints_to_label += [p + adjustment for p in timepoints_to_label[i]]
        pca_points_at_timestamps = np.array([pca_components[:, i] for i in range(pca_components.shape[1]) if i in
                                             flattened_timepoints_to_label])
        try:
            if n_components == 2:
                plt.scatter(pca_points_at_timestamps[:, 0], pca_points_at_timestamps[:, 1], marker="x", color="r")
            elif n_components == 3:
                ax.scatter(pca_points_at_timestamps[:, 0], pca_points_at_timestamps[:, 1],
                           pca_points_at_timestamps[:, 2], marker="x", color="r")
        except IndexError:
            pass

    if n_components == 2:
        plt.colorbar()

    plt.title("PCA Trajectory Space: " + context_name)
    if exclude_outliers:
        mu, sigma, min_threshold, max_threshold = estimate_gaussian(pca_components[0])
        mu, sigma, min_threshold2, max_threshold2 = estimate_gaussian(pca_components[1])
        plt.xlim(min_threshold, max_threshold)
        plt.ylim(min_threshold2, max_threshold2)
    plt.show()


def plot_pca_trajectory_multiple_trials_environmental_position(activity_data, fish_position_data, display_numbers=True,
                                                               context_name="No Label", self_normalise_activity_data=True):
    flattened_activity_data = np.concatenate((activity_data), axis=1)
    if self_normalise_activity_data:
        flattened_activity_data = normalise_within_neuron_multiple_traces(flattened_activity_data)

    pca = PCA(n_components=2)
    pca.fit(flattened_activity_data)
    pca_components = pca.components_[:, :]

    split_colours = np.array([])
    for i in range(len(activity_data)):
        split_colours = np.concatenate((split_colours, fish_position_data[i][:, 0].astype(int)))

    fig, ax = plt.subplots(figsize=(10, 10))
    if display_numbers:
        for i in range(len(pca_components[0])):
            ax.annotate(i, (pca_components[0, i], pca_components[1, i]))

    plt.scatter(pca_components[0], pca_components[1], c=split_colours)
    plt.colorbar()
    plt.title(context_name)
    plt.show()


    split_colours = np.array([])
    for i in range(len(activity_data)):
        split_colours = np.concatenate((split_colours, fish_position_data[i][:, 1].astype(int)))

    fig, ax = plt.subplots(figsize=(10, 10))
    if display_numbers:
        for i in range(len(pca_components[0])):
            ax.annotate(i, (pca_components[0, i], pca_components[1, i]))

    plt.scatter(pca_components[0], pca_components[1], c=split_colours)
    plt.colorbar()
    plt.title(context_name)
    plt.show()


if __name__ == "__main__":
    rnn_data_full = []
    fish_position_data = []
    consumption_points = []
    datas = []
    for i in range(1, 6):
        data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Endless", f"Naturalistic-{i}")
        # data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
        rnn_data = data["rnn_state_actor"][:, 0, 0, :]
        rnn_data = np.swapaxes(rnn_data, 0, 1)
        consumption_points.append([i for i in range(len(data["consumed"][:])) if data["consumed"][i]])
        rnn_data_full.append(rnn_data)
        fish_position_data.append(data["fish_position"])
        datas.append(data)

    # plot_pca_trajectory_multiple_trials_environmental_position(rnn_data_full, fish_position_data, display_numbers=False)
    # behavioural_labels = label_behavioural_context_multiple_trials(datas, model_name="dqn_scaffold_18-1")
    # consumption_points = [[i for i, b in enumerate(be[:, 6]) if b == 1] for be in behavioural_labels]
    # consumption_points = []
    # plot_pca_trajectory(rnn_data_full, timepoints_to_label=consumption_points)
    plot_pca_trajectory_multiple_trials(rnn_data_full, consumption_points, display_numbers=False)
    # data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Endless", f"Naturalistic-1")
    # rnn_data = np.swapaxes(data["rnn_state_actor"][:, 0, 0, :], 0, 1)
    # # reduced_rnn_data = remove_those_with_no_output(rnn_data, "dqn_scaffold_18-2", "dqn_18_2", proportion_to_remove=0.2)
    # reduced_rnn_data = rnn_data[:256]
    # # reduced_rnn_data = rnn_data[256:]
    # # reduced_rnn_data = rnn_data
    #
    # plot_pca_trajectory(reduced_rnn_data, timepoints_to_label=None)
    # positions = data["fish_position"][3000:]
    # plt.scatter(positions[:, 0], positions[:, 1])
    # plt.show()

