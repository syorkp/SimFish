import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

from Analysis.load_data import load_data

from Analysis.Behavioural.Tools.label_behavioural_context import label_behavioural_context_multiple_trials,\
    get_behavioural_context_name_by_index
from Analysis.Neural.Systems.plot_pca_trajectory import plot_pca_trajectory, plot_pca_trajectory_multiple_trials


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
    x = np.concatenate((associated_periods))
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

    plt.scatter(pca_components[0], pca_components[1], c=prevailing_context_full, alpha=0.1)
    plt.legend(handles=legend_elements)
    plt.show()


def plot_pca_with_all_behavioural_periods_multiple_trials(datas, display_timestamps=False):
    rnn_data_full = []
    consumption_points = []
    for data in datas:
        rnn_data = data["rnn_state_actor"][:, 0, 0, :]
        rnn_data = np.swapaxes(rnn_data, 0, 1)
        consumption_points.append([i for i in range(len(data["consumed"][:])) if data["consumed"][i]])
        rnn_data_full.append(rnn_data)
    behavioural_labels = label_behavioural_context_multiple_trials(datas, model_name="dqn_scaffold_18-2")

    for behaviour in range(behavioural_labels[0].shape[1]):
        behavioural_points = [[i for i, b in enumerate(be[:, behaviour]) if b == 1] for be in behavioural_labels]
        label_name = get_behavioural_context_name_by_index(behaviour)
        plot_pca_trajectory_multiple_trials(rnn_data_full, behavioural_points, context_name=label_name,
                                            display_numbers=display_timestamps)


if __name__ == "__main__":
    datas = []
    for i in range(1, 2):
        data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", f"Naturalistic-{i}")
    # for i in range(1, 11):
    #     data = load_data("dqn_scaffold_18-2", "Behavioural-Data-Free", f"Naturalistic-{i}")
        datas.append(data)
    plot_pca_with_all_behavioural_periods_multiple_trials(datas, display_timestamps=True)
    # plot_pca_trajectory_with_contexts_multiple_trials(datas, remove_value_stream=False)







