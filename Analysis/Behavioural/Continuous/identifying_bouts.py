
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from Analysis.Behavioural.Tools.BehavLabels.label_behavioural_context import label_capture_sequences, label_exploration_sequences_free_swimming
from Analysis.Behavioural.Continuous.plot_behavioural_choice import get_multiple_means
from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.show_action_sequence_block import display_all_sequences


def cluster_bouts(impulses, angles, cluster_type, cluster_num, model_name, impulse_scaling, angle_scaling,
                  angle_absolute=True, return_actions=False):
    """Returns index labels for all impulse-angle pairs provided."""
    if angle_absolute:
        angles_a = np.absolute(angles)
    else:
        angles_a = angles

    # for ep in np.linspace(0.0001, 0.05, 10):
    if cluster_type == "KNN":
        cluster_obj = KMeans(n_clusters=cluster_num, n_init=20)
        start_i = 0
    elif cluster_type == "AGG":
        cluster_obj = AgglomerativeClustering(n_clusters=cluster_num)
        start_i = 0
    elif cluster_type == "DBSCAN":
        cluster_obj = DBSCAN(eps=ep, min_samples=80)
        start_i = -1
    else:
        cluster_obj = None
        start_i = 0
    nbrs = cluster_obj.fit([[imp, ang] for imp, ang in zip(impulses, angles_a)])
    labels_new = nbrs.labels_
    cluster_num = len(set(labels_new))
    print(cluster_num)

    # if cluster_num > 7 or cluster_num == 1:
    #     continue
    colours = ["b", "g", "r", "y", "c", "m", "black"]
    ind = labels_new.astype(int)
    for a in range(start_i, cluster_num-start_i):
        plt.scatter(impulses[labels_new == a] * impulse_scaling, angles[labels_new == a] * angle_scaling, c=colours[a])
    plt.legend([str(a) for a in range(cluster_num)])
    plt.xlabel("Impulse")
    plt.ylabel("Angle")
    plt.savefig(f"All-Plots/{model_name}/{cluster_type}-{cluster_num}-clustered.jpg")
    plt.clf()
    plt.close()

    if return_actions:
        impulses *= impulse_scaling
        angles *= angle_scaling

        return nbrs, np.concatenate((np.expand_dims(impulses, 1), np.expand_dims(angles, 1)), axis=1)
    else:
        return nbrs


def get_ppo_prey_capture_seq(model_name, assay_config, assay_id, n, predictor, associated_actions):
    """Predictor object contains labels for all sequences - just have to find new standardised way of accessing."""
    prey_capture_sequences = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        capture_ts = label_capture_sequences(data, n=20) * 1

        capture_ts_separated = []
        current_ts_sequence = []
        for i, c in enumerate(capture_ts):
            if c == 1:
                current_ts_sequence.append(i)
            else:
                if len(current_ts_sequence) > 0:
                    capture_ts_separated.append(current_ts_sequence)
                    current_ts_sequence = []
        if len(current_ts_sequence) > 0:
            capture_ts_separated.append(current_ts_sequence)
            current_ts_sequence = []

        for sequence in capture_ts_separated:
            impulses_separated = data["impulse"][sequence]
            angles_separated = data["angle"][sequence]
            index_i = np.where(associated_actions[:, 0] == impulses_separated[0])
            index_a = np.where(associated_actions[:, 1] == angles_separated[0])
            if index_i[0] == index_a[0]:
                actions = predictor.labels_[int(index_i[0]):int(index_i[0])+len(sequence)]
            else:
                print("Matching error...")
                actions = []
            prey_capture_sequences.append(actions)

    return prey_capture_sequences


def get_ppo_exploration_seq(model_name, assay_config, assay_id, n, predictor, associated_actions):
    prey_capture_sequences = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        capture_ts = label_exploration_sequences_free_swimming(data) * 1

        capture_ts_separated = []
        current_ts_sequence = []
        for i, c in enumerate(capture_ts):
            if c == 1:
                current_ts_sequence.append(i)
            else:
                if len(current_ts_sequence) > 0:
                    capture_ts_separated.append(current_ts_sequence)
                    current_ts_sequence = []
        if len(current_ts_sequence) > 0:
            capture_ts_separated.append(current_ts_sequence)

        for sequence in capture_ts_separated:
            impulses_separated = data["impulse"][sequence]
            angles_separated = data["angle"][sequence]
            index_i = np.where(associated_actions[:, 0] == impulses_separated[0])
            index_a = np.where(associated_actions[:, 1] == angles_separated[0])

            if len(impulses_separated) > 4:
                if index_i[0] == index_a[0]:
                    actions = predictor.labels_[int(index_i[0]):int(index_i[0]) + len(sequence)]
                else:
                    print("Matching error...")
                    actions = []
                prey_capture_sequences.append(actions)

    return prey_capture_sequences


def get_ppo_spatial_density_plots(model_name, assay_config, assay_id, n, predictor):
    ...


if __name__ == "__main__":
    model_name, assay_config, assay_id, n = "ppo_scaffold_21-2", "Behavioural-Data-Free", "Naturalistic", 20
    assay_config_emtpy = "Behavioural-Data-Empty"
    n_clusters = 5
    mu_impulse, mu_angle = get_multiple_means(model_name, assay_config, assay_id, n)
    mu_impulse_empty, mu_angle_empty = get_multiple_means(model_name, assay_config_emtpy, assay_id, n)
    mu_impulse = np.concatenate((mu_impulse, mu_impulse_empty))
    mu_angle = np.concatenate((mu_angle, mu_angle_empty))

    p, associated_bouts = cluster_bouts(mu_impulse, mu_angle, "KNN", n_clusters, model_name, 16, 1, return_actions=True)

    seqs = get_ppo_prey_capture_seq(model_name, assay_config, assay_id, n, predictor=p, associated_actions=associated_bouts)
    display_all_sequences(seqs, max_length=20, alternate_action_names=[str(i) for i in range(n_clusters)])

    seqs = get_ppo_exploration_seq(model_name, "Behavioural-Data-Empty", assay_id, n, predictor=p, associated_actions=associated_bouts)
    display_all_sequences(seqs, max_length=100, alternate_action_names=[str(i) for i in range(n_clusters)])

    # sil = []
    # for i in range(2, 10):
    #     nbrs = KMeans(n_clusters=i, n_init=20).fit([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)])
    #     labels_new = nbrs.labels_
    #     sil.append(silhouette_score([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)], labels_new, metric='euclidean'))
    #
    # optimal_num = sil.index(min(sil)) + 1
    # nbrs = KMeans(n_clusters=optimal_num, n_init=20).fit([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)])
    # labels_new = nbrs.labels_

    # sil = []
    # for i in range(2, 10):
    #     nbrs = AgglomerativeClustering(n_clusters=i).fit([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)])
    #     labels_new = nbrs.labels_
    #     sil.append(silhouette_score([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)], labels_new, metric='euclidean'))
    #
    # optimal_num = sil.index(min(sil)) + 1
    # nbrs = AgglomerativeClustering(n_clusters=2).fit([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)])
    # labels_new = nbrs.labels_


