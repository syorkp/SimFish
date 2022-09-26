
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from Analysis.Behavioural.Continuous.plot_behavioural_choice import get_multiple_means


if __name__ == "__main__":
    model_name, assay_config, assay_id, n = "ppo_scaffold_21-2", "Behavioural-Data-Free", "Naturalistic", 10
    mu_impulse, mu_angle = get_multiple_means(model_name, assay_config, assay_id, n)
    mu_angle = np.absolute(mu_angle)

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
    # nbrs = AgglomerativeClustering(n_clusters=optimal_num).fit([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)])
    # labels_new = nbrs.labels_

    sil = []
    for i in range(2, 10):
        nbrs = DBSCAN(eps=0.8125, min_samples=2).fit([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)])
        labels_new = nbrs.labels_
        sil.append(silhouette_score([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)], labels_new, metric='euclidean'))

    optimal_num = sil.index(min(sil)) + 1
    nbrs = DBSCAN(eps=0.8125, min_samples=5).fit([[f, ifc_v] for f, ifc_v in zip(mu_impulse, mu_angle)])
    labels_new = nbrs.labels_

    plt.scatter(mu_impulse, mu_angle, c=labels_new)
    plt.show()
