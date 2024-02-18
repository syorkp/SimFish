import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy import signal

from Analysis.load_data import load_data

from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces


def plot_pca_trajectory(activity_data, timepoints_to_label=None):
    # step_data = activity_data[:, t:t+1]
    pca = PCA(n_components=2)
    pca.fit(activity_data)
    pca_components = pca.components_
    plt.scatter(pca_components[0], pca_components[1], c=np.arange(len(pca_components[0])))
    if timepoints_to_label is not None:
        pca_points_at_timestamps = np.array(
            [pca_components[:, i] for i in range(pca_components.shape[1]) if i in timepoints_to_label])
        plt.scatter(pca_points_at_timestamps[:, 0], pca_points_at_timestamps[:, 1], marker="x", color="r")
    plt.colorbar()
    plt.show()


def estimate_gaussian(dataset):
    mu = np.mean(dataset)  # moyenne cf mu
    sigma = np.std(dataset)  # écart_type/standard deviation
    limit = sigma * 10

    min_threshold = mu - limit
    max_threshold = mu + limit

    return mu, sigma, min_threshold, max_threshold


def plot_pca_trajectory_multiple_trials(activity_data, timepoints_to_label=None, display_numbers=True,
                                        context_name="No Label", self_normalise_activity_data=True, n_components=2,
                                        exclude_outliers=True, detrend=True,
                                        include_only_active_neurons=True):
    flattened_activity_data = np.concatenate((activity_data), axis=1)
    if self_normalise_activity_data:
        flattened_activity_data = normalise_within_neuron_multiple_traces(flattened_activity_data)

    if include_only_active_neurons:
        std_main = np.std(flattened_activity_data, axis=1)
        varying_main = (std_main > 0.4)
        n_neurons = np.sum(varying_main * 1)
        flattened_activity_data = flattened_activity_data[varying_main]
    else:
        n_neurons = flattened_activity_data.shape[0]
    if detrend:
        flattened_activity_data = signal.detrend(flattened_activity_data)

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
            adjustment = np.sum(len_of_each[:i + 1])
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

    plt.title(f"PCA Phase Space: {context_name}  Neurons included: {n_neurons}")
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
            adjustment = np.sum(len_of_each[:i + 1])
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
                                                               context_name="No Label",
                                                               self_normalise_activity_data=True):
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


def plot_pca_directly(pca_components, activity_data, timepoints_to_label, n_components, context_name,
                      exclude_outliers=False, plot_name="No Name", alph=0.05):
    # Phase space
    fig, ax = plt.subplots(figsize=(10, 10))

    len_of_each = [0] + [len(c[0]) for c in activity_data][:-1]
    flattened_timepoints_to_label = []
    for i, c in enumerate(timepoints_to_label):
        adjustment = np.sum(len_of_each[:i + 1])
        flattened_timepoints_to_label += [p + adjustment for p in timepoints_to_label[i]]

    if n_components == 2:
        plt.scatter(pca_components[0], pca_components[1], alpha=alph)
        plt.scatter(pca_components[0, flattened_timepoints_to_label], pca_components[1, flattened_timepoints_to_label],
                    alpha=alph, color="r")
    elif n_components == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pca_components[0], pca_components[1], pca_components[2], alpha=alph)
        ax.scatter(pca_components[0, flattened_timepoints_to_label],
                   pca_components[1, flattened_timepoints_to_label],
                   pca_components[2, flattened_timepoints_to_label], alpha=alph, color="r")
    else:
        print("Unsupported number of components")

    plt.title(f"PCA {plot_name}: " + context_name)

    if exclude_outliers:
        mu, sigma, min_threshold, max_threshold = estimate_gaussian(pca_components[0])
        mu, sigma, min_threshold2, max_threshold2 = estimate_gaussian(pca_components[1])
        plt.xlim(min_threshold, max_threshold)
        plt.ylim(min_threshold2, max_threshold2)
    plt.show()


def plot_pca_directly_all_behaviours(pca_components, activity_data, behavioural_labels, n_components,
                                     selected_behaviours, exclude_outliers=False, plot_name="No Name", alph=0.05):
    # Phase space
    fig, ax = plt.subplots(figsize=(10, 10))
    flattened_behavioural_labels = np.concatenate((behavioural_labels), axis=0)

    flattened_behavioural_labels = flattened_behavioural_labels[:, selected_behaviours]
    plt.scatter(pca_components[0], pca_components[1],
                alpha=alph, color="grey")
    colours = ["b", "r", "g", "k", "l"]
    for behaviour in range(flattened_behavioural_labels.shape[1]):
        current_labels = flattened_behavioural_labels[:, behaviour]
        timestamps = np.array([i for i, l in enumerate(current_labels) if l == 1])
        if n_components == 2:
            plt.scatter(pca_components[0, timestamps], pca_components[1, timestamps],
                        alpha=alph, color=colours[behaviour])
        elif n_components == 3:
            print("Not implemented")
        else:
            print("Unsupported number of components")

    plt.title(f"PCA {plot_name}: ")

    if exclude_outliers:
        mu, sigma, min_threshold, max_threshold = estimate_gaussian(pca_components[0])
        mu, sigma, min_threshold2, max_threshold2 = estimate_gaussian(pca_components[1])
        plt.xlim(min_threshold, max_threshold)
        plt.ylim(min_threshold2, max_threshold2)
    plt.show()


def plot_pca_directly_hist(pca_components, activity_data, timepoints_to_label, n_components, context_name,
                           exclude_outliers=False, plot_name="No Name", alph=0.05, valid_threshold=10):


    len_of_each = [0] + [len(c[0]) for c in activity_data][:-1]
    flattened_timepoints_to_label = []
    for i, c in enumerate(timepoints_to_label):
        adjustment = np.sum(len_of_each[:i + 1])
        flattened_timepoints_to_label += [p + adjustment for p in timepoints_to_label[i]]

    if len(flattened_timepoints_to_label) == 0:
        return


    if exclude_outliers:
        mu, sigma, min_threshold, max_threshold = estimate_gaussian(pca_components[0])
        mu, sigma, min_threshold2, max_threshold2 = estimate_gaussian(pca_components[1])
        pca_to_exclude = (pca_components[0, :] < min_threshold) + (pca_components[0, :] > max_threshold) + \
                         (pca_components[1, :] < min_threshold2) + (pca_components[1, :] > max_threshold2)
        pca_components = pca_components[:, ~pca_to_exclude]
        timepoints_to_exclude = [i for i, b in enumerate(pca_to_exclude) if b]

        actual_excluded_points = [t for t in timepoints_to_exclude if t in flattened_timepoints_to_label]
        flattened_timepoint_index = 0
        new_flattened_timepoints_to_label = []
        current_timepoint = flattened_timepoints_to_label[flattened_timepoint_index]
        for i, t in enumerate(actual_excluded_points):
            while current_timepoint < t:
                new_flattened_timepoints_to_label.append(current_timepoint-(i))
                flattened_timepoint_index += 1
                current_timepoint = flattened_timepoints_to_label[flattened_timepoint_index]
            flattened_timepoint_index += 1
            current_timepoint = flattened_timepoints_to_label[flattened_timepoint_index]

        # flattened_timepoint_index -= 1
        # for c in range(flattened_timepoint_index+1, len(flattened_timepoints_to_label)):
        #     new_flattened_timepoints_to_label.append(flattened_timepoints_to_label[c] - (i+1))

        new_flattened_timepoints_to_label = np.array(new_flattened_timepoints_to_label)
        # flattened_timepoints_to_label = np.array(flattened_timepoints_to_label)
        flattened_timepoints_to_label = new_flattened_timepoints_to_label

    # Phase space
    fig, ax = plt.subplots(figsize=(10, 10))

    if n_components == 2:
        all_points_hist = np.histogram2d(pca_components[0], pca_components[1], bins=100)

        if len(flattened_timepoints_to_label) == 0:
            return
        selected_points_hist = np.histogram2d(pca_components[0, flattened_timepoints_to_label],
                                              pca_components[1, flattened_timepoints_to_label],
                                              bins=[all_points_hist[1], all_points_hist[2]])[0]



        hist_proportions = selected_points_hist / all_points_hist[0]
        hist_proportions[np.isnan(hist_proportions)] = 0
        normal_proportions = all_points_hist[0] / np.max(all_points_hist[0])

        few_present = all_points_hist[0] < valid_threshold

        hist_proportions = np.expand_dims(hist_proportions, 2)
        normal_proportions = np.expand_dims(normal_proportions, 2)
        coloured = np.concatenate((hist_proportions, np.zeros(hist_proportions.shape), normal_proportions), axis=2)

        coloured[few_present] = 0
        plt.imshow(coloured)

        # Can also have conditional to only show areas where there is a high enough density...


    elif n_components == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pca_components[0], pca_components[1], pca_components[2], alpha=alph)
        ax.scatter(pca_components[0, flattened_timepoints_to_label],
                   pca_components[1, flattened_timepoints_to_label],
                   pca_components[2, flattened_timepoints_to_label], alpha=alph, color="r")
    else:
        print("Unsupported number of components")

    plt.title(f"PCA {plot_name}: " + context_name)
    plt.show()


def plot_pca_directly_hist_all_behaviours(pca_components, activity_data, behavioural_labels, n_components,
                                     selected_behaviours, exclude_outliers=False, plot_name="No Name", valid_threshold=10):


    flattened_behavioural_labels = np.concatenate((behavioural_labels), axis=0)
    flattened_behavioural_labels = flattened_behavioural_labels[:, selected_behaviours]

    if exclude_outliers:
        mu, sigma, min_threshold, max_threshold = estimate_gaussian(pca_components[0])
        mu, sigma, min_threshold2, max_threshold2 = estimate_gaussian(pca_components[1])
        pca_to_exclude = (pca_components[0, :] < min_threshold) + (pca_components[0, :] > max_threshold) + \
                         (pca_components[1, :] < min_threshold2) + (pca_components[1, :] > max_threshold2)
        pca_components = pca_components[:, ~pca_to_exclude]
        flattened_behavioural_labels = flattened_behavioural_labels[~pca_to_exclude, :]

    # Phase space
    fig, ax = plt.subplots(figsize=(10, 10))

    colour_groups = [(1, 0, 0), (0, 1, 0)]  # No behaviour appears as blue.

    if n_components == 2:
        all_points_hist = np.histogram2d(pca_components[0], pca_components[1], bins=100)

        normal_proportions = all_points_hist[0] / np.max(all_points_hist[0])
        normal_proportions = np.expand_dims(normal_proportions, 2)
        few_present = all_points_hist[0] < valid_threshold

        additional_dims = np.zeros(all_points_hist[0].shape)
        additional_dims = np.expand_dims(additional_dims, 2)
        coloured = np.concatenate((additional_dims, additional_dims, normal_proportions), 2)

        for behaviour in range(flattened_behavioural_labels.shape[1]):
            behavioural_points = flattened_behavioural_labels[:, behaviour].astype(int)
            behavioural_points = [i for i, p in enumerate(behavioural_points) if p == 1]

            selected_points_x = pca_components[0, behavioural_points]
            selected_points_y = pca_components[1, behavioural_points]
            selected_points_hist = np.histogram2d(selected_points_x,
                                                  selected_points_y,
                                                  bins=[all_points_hist[1], all_points_hist[2]])
            selected_points_hist = selected_points_hist[0]


            hist_proportions = selected_points_hist / all_points_hist[0]
            hist_proportions[np.isnan(hist_proportions)] = 0
            hist_proportions = np.expand_dims(hist_proportions, 2)
            hist_proportions = np.repeat(hist_proportions, 3, 2)

            hist_proportions = hist_proportions * colour_groups[behaviour]
            coloured += hist_proportions

        coloured[few_present] = 0
        plt.imshow(coloured)

        # Can also have conditional to only show areas where there is a high enough density...


    elif n_components == 3:
        print("Not Implemented")
    else:
        print("Unsupported number of components")

    plt.title(f"PCA {plot_name}: ")
    plt.show()



if __name__ == "__main__":
    rnn_data_full = []
    fish_position_data = []
    consumption_points = []
    datas = []
    choices = [3, 5]
    for i in choices:
        data = load_data("dqn_scaffold_14-1", "Interruptions-HA", f"Naturalistic-{i}")
        # data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
        rnn_data = data["rnn_state"][:, 0, 0, :]
        rnn_data = np.swapaxes(rnn_data, 0, 1)
        consumption_points.append([i for i in range(len(data["consumed"][:])) if data["consumed"][i]])
        rnn_data_full.append(rnn_data)
        fish_position_data.append(data["fish_position"])
        datas.append(data)

    consumption_points = []
    # plot_pca_trajectory_multiple_trials_environmental_position(rnn_data_full, fish_position_data, display_numbers=False)
    # behavioural_labels = label_behavioural_context_multiple_trials(datas, model_name="dqn_scaffold_18-1")
    # consumption_points = [[i for i, b in enumerate(be[:, 6]) if b == 1] for be in behavioural_labels]
    # consumption_points = []
    # plot_pca_trajectory(rnn_data_full, timepoints_to_label=consumption_points)
    plot_pca_trajectory_multiple_trials(rnn_data_full, consumption_points, display_numbers=False, detrend=True,
                                        include_only_active_neurons=True)
    # data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Endless", f"Naturalistic-1")
    # rnn_data = np.swapaxes(data["rnn_state"][:, 0, 0, :], 0, 1)
    # # reduced_rnn_data = remove_those_with_no_output(rnn_data, "dqn_scaffold_18-2", "dqn_18_2", proportion_to_remove=0.2)
    # reduced_rnn_data = rnn_data[:256]
    # # reduced_rnn_data = rnn_data[256:]
    # # reduced_rnn_data = rnn_data
    #
    # plot_pca_trajectory(reduced_rnn_data, timepoints_to_label=None)
    # positions = data["fish_position"][3000:]
    # plt.scatter(positions[:, 0], positions[:, 1])
    # plt.show()
