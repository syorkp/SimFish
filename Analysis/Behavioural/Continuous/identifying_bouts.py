
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from Analysis.Behavioural.Tools.BehavLabels.label_behavioural_context import label_capture_sequences, label_exploration_sequences_free_swimming
from Analysis.Behavioural.Continuous.plot_behavioural_choice import get_multiple_means
from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.show_action_sequence_block import display_all_sequences


def cluster_bouts(impulses, angles, cluster_type, cluster_num, model_name, angle_absolute=True):
    if angle_absolute:
        angles = np.absolute(angles)

    if cluster_type == "KNN":
        cluster_obj = KMeans(n_clusters=cluster_num, n_init=20)
    elif cluster_type == "AGG":
        cluster_obj = AgglomerativeClustering(n_clusters=cluster_num)
    else:
        cluster_obj = None

    nbrs = cluster_obj.fit([[imp, ang] for imp, ang in zip(impulses, angles)])
    labels_new = nbrs.labels_

    scatter = plt.scatter(mu_impulse, mu_angle, c=labels_new)
    plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(max(labels_new + 1))])
    plt.savefig(f"All-Plots/{model_name}/{cluster_type}-{cluster_num}-clustered.jpg")
    plt.clf()

    return nbrs


def get_ppo_prey_capture_seq(model_name, assay_config, assay_id, n, predictor):
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
            actions = predictor.fit_predict([[imp, ang] for imp, ang in zip(impulses_separated, angles_separated)])
            prey_capture_sequences.append(actions)

    return prey_capture_sequences


def get_ppo_exploration_seq(model_name, assay_config, assay_id, n, predictor):
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
            if len(impulses_separated) > 4:
                actions = predictor.predict([[imp, ang] for imp, ang in zip(impulses_separated, angles_separated)])
                prey_capture_sequences.append(actions)

    return prey_capture_sequences


if __name__ == "__main__":
    model_name, assay_config, assay_id, n = "ppo_scaffold_21-2", "Behavioural-Data-Free", "Naturalistic", 20
    mu_impulse, mu_angle = get_multiple_means(model_name, assay_config, assay_id, n)
    p = cluster_bouts(mu_impulse, mu_angle, "KNN", 5, model_name)
    seqs = get_ppo_prey_capture_seq(model_name, assay_config, assay_id, n, predictor=p)
    display_all_sequences(seqs, max_length=20)

    seqs = get_ppo_exploration_seq(model_name, "Behavioural-Data-Empty", assay_id, n, predictor=p)
    display_all_sequences(seqs, max_length=100)

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

