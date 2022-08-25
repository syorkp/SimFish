import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

from Analysis.load_data import load_data

from Analysis.Behavioural.Tools.label_behavioural_context import label_behavioural_context_multiple_trials,\
    get_behavioural_context_name_by_index
from Analysis.Neural.Systems.plot_pca_trajectory import plot_pca_trajectory, plot_pca_trajectory_multiple_trials, \
    plot_pca_directly, plot_pca_directly_hist, plot_pca_directly_all_behaviours
from Analysis.Neural.Regression.label_cell_roles import get_category_indices
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces


def plot_pca_trajectory_with_contexts_multiple_trials(datas, remove_value_stream=False):
    behavioural_labels = label_behavioural_context_multiple_trials(datas, environment_size=1500)
    if remove_value_stream:
        all_activity_data = [np.swapaxes(data["rnn_state_actor"][:, 0, 0, :256], 0, 1) for data in datas]
    else:
        all_activity_data = [np.swapaxes(data["rnn_state_actor"][:, 0, 0, :], 0, 1) for data in datas]
    plot_pca_trajectory_with_context(all_activity_data, behavioural_labels)


def plot_pca_trajectory_with_context(activity_data, associated_periods):
    flattened_activity_data = np.concatenate((activity_data), axis=1)
    pca = PCA(n_components=2)
    pca.fit(flattened_activity_data)
    pca_components = pca.components_

    split_colours = np.array([])
    for i in range(len(activity_data)):
        split_colours = np.concatenate((split_colours, np.arange(len(activity_data[i][0]))))

    prevailing_context_full = []
    for trial in associated_periods:
        for step in trial:
            if 3 in step:
                prevailing_context_full.append("black")
            else:
                if 1 in step:
                    prevailing_context_full.append("y")
                else:
                    if 2 in step or 4 in step:
                        prevailing_context_full.append("g")
                    else:
                        if 5 in step:
                            prevailing_context_full.append("r")
                        else:
                            prevailing_context_full.append("blue")

    legend_elements = [Line2D([0], [0], marker='o', color='blue', label='None',
                              markerfacecolor='blue', markersize=15),
                       Line2D([0], [1], marker='o', color='y', label='Prey Capture',
                              markerfacecolor='y', markersize=15),
                       Line2D([0], [2], marker='o', color='g', label='Exploration-FS',
                              markerfacecolor='g', markersize=15),
                       Line2D([0], [3], marker='o', color='black', label='Avoidance',
                              markerfacecolor='black', markersize=15),
                       Line2D([0], [4], marker='o', color='r', label='Wall',
                              markerfacecolor='r', markersize=15),
                       ]

    plt.title("PCA Phase Space")
    plt.scatter(pca_components[0], pca_components[1], c=prevailing_context_full, alpha=0.1)
    plt.legend(handles=legend_elements)
    plt.show()

    plt.title("PCA Trajectory Space")
    pca_trajectory = pca_components[:, 1:] - pca_components[:, :-1]
    plt.scatter(pca_trajectory[0], pca_trajectory[1], c=prevailing_context_full, alpha=0.1)
    plt.legend(handles=legend_elements)
    plt.show()


def plot_pca_with_all_behavioural_periods_multiple_trials(datas, model_name, display_timestamps=False,
                                                          remove_value_stream=False, n_components=2,
                                                          selected_neurons=None):
    rnn_data_full = []
    consumption_points = []
    for data in datas:
        if selected_neurons is None:
            if remove_value_stream:
                rnn_data = data["rnn_state_actor"][:, 0, 0, :256]
            else:
                rnn_data = data["rnn_state_actor"][:, 0, 0, :]
        else:
            if remove_value_stream:
                selected_neurons = [i for i in selected_neurons if i < 256]
            rnn_data = data["rnn_state_actor"][:, 0, 0, selected_neurons]

        rnn_data = np.swapaxes(rnn_data, 0, 1)
        consumption_points.append([i for i in range(len(data["consumed"][:])) if data["consumed"][i]])
        rnn_data_full.append(rnn_data)
    behavioural_labels = label_behavioural_context_multiple_trials(datas, model_name=model_name)

    for behaviour in range(behavioural_labels[0].shape[1]):
        behavioural_points = [[i for i, b in enumerate(be[:, behaviour]) if b == 1] for be in behavioural_labels]
        label_name = get_behavioural_context_name_by_index(behaviour)
        plot_pca_trajectory_multiple_trials(rnn_data_full, behavioural_points, context_name=label_name,
                                            display_numbers=display_timestamps, n_components=n_components)


def plot_pca_with_all_behavioural_periods_multiple_trials_2(datas, model_name, display_timestamps=False,
                                                          remove_value_stream=False, n_components=2,
                                                          selected_neurons=None, self_normalise_activity_data=True):
    """Second version of previous, with difference being - is more efficient in that computes PCA once only, then plots
    behavioural contexts as colours (to allow distinguishing relative density), rather than crosses.
    """
    rnn_data_full = []
    consumption_points = []
    for data in datas:
        if selected_neurons is None:
            if remove_value_stream:
                rnn_data = data["rnn_state_actor"][:, 0, 0, :256]
            else:
                rnn_data = data["rnn_state_actor"][:, 0, 0, :]
        else:
            if remove_value_stream:
                selected_neurons = [i for i in selected_neurons if i < 256]
            rnn_data = data["rnn_state_actor"][:, 0, 0, selected_neurons]

        rnn_data = np.swapaxes(rnn_data, 0, 1)
        consumption_points.append([i for i in range(len(data["consumed"][:])) if data["consumed"][i]])
        rnn_data_full.append(rnn_data)
    behavioural_labels = label_behavioural_context_multiple_trials(datas, model_name=model_name)

    # Do PCA
    flattened_activity_data = np.concatenate((rnn_data_full), axis=1)
    if self_normalise_activity_data:
        flattened_activity_data = normalise_within_neuron_multiple_traces(flattened_activity_data)

    pca = PCA(n_components=n_components)
    pca.fit(flattened_activity_data)
    pca_components = pca.components_[:, :]
    pca_components_trajectory = pca_components[:, 1:] - pca_components[:, :-1]
    pca_components_trajectory = np.concatenate((pca_components_trajectory[:, 0:1], pca_components_trajectory), axis=1)

    # for behaviour in range(behavioural_labels[0].shape[1]):
    #     behavioural_points = [[i for i, b in enumerate(be[:, behaviour]) if b == 1] for be in behavioural_labels]
    #     label_name = get_behavioural_context_name_by_index(behaviour)
    #     plot_pca_directly_hist(pca_components, rnn_data_full, behavioural_points, context_name=label_name,
    #                       n_components=n_components, plot_name="Phase Space")
    #     plot_pca_directly(pca_components, rnn_data_full, behavioural_points, context_name=label_name,
    #                       n_components=n_components, plot_name="Phase Space")
    #     plot_pca_directly_hist(pca_components_trajectory, rnn_data_full, behavioural_points, context_name=label_name,
    #                            n_components=n_components, plot_name="Trajectory Space", exclude_outliers=True)
    #     plot_pca_directly(pca_components_trajectory, rnn_data_full, behavioural_points, context_name=label_name,
    #                       n_components=n_components, plot_name="Trajectory Space", exclude_outliers=True)


    behav_indices = [5, 9]  # Only show a few of the conditions, otherwise is overwhelemed by common contexts.
    plot_pca_directly_all_behaviours(pca_components, rnn_data_full, behavioural_labels, n_components=n_components,
                                     plot_name="Phase Space", alph=0.01, selected_behaviours=behav_indices)
    plot_pca_directly_all_behaviours(pca_components_trajectory, rnn_data_full, behavioural_labels, n_components=n_components,
                                     plot_name="Phase Space", alph=0.01, selected_behaviours=behav_indices, exclude_outliers=True)

if __name__ == "__main__":
    datas = []
    model_name = "dqn_scaffold_18-1"
    # for i in range(1, 2):
    #     data = load_data(model_name, "Behavioural-Data-Free", f"Naturalistic-{i}")
    for i in range(1, 11):
        data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Endless", f"Naturalistic-{i}")
        datas.append(data)

    # energy_state_neurons = get_category_indices("dqn_scaffold_18-1", "Behavioural-Data-Endless", "Naturalistic", 3,
    #                                             "Starving", score_threshold=0.2)

    plot_pca_with_all_behavioural_periods_multiple_trials_2(datas, model_name, display_timestamps=False,
                                                          remove_value_stream=True, n_components=2)
    # plot_pca_with_all_behavioural_periods_multiple_trials(datas, model_name, display_timestamps=False,
    #                                                       remove_value_stream=True, n_components=2,
    #                                                       selected_neurons=energy_state_neurons)
    # plot_pca_trajectory_with_contexts_multiple_trials(datas, remove_value_stream=False)







