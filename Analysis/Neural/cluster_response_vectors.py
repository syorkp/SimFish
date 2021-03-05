import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data
from Analysis.Neural.calculate_vrv import get_all_neuron_vectors, get_stimulus_vector, normalise_vrvs


def plot_tsne_results(vectors, title="None"):
    tsne = TSNE(n_components=2, n_iter=400, perplexity=50)
    tsne_results = tsne.fit_transform(vectors)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    tpd['Point'] = ["Blue" for i in range(len(tsne_results[:, 0]))]
    tpd["Point"][0] = "Red"
    plt.figure(figsize=(16, 10))
    plt.title(title)

    p1 = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # palette=sns.color_palette("hls", 10),
        hue="Point",
        # palette=sns.color_palette("hls"),
        data=tpd,
        legend="full",
        alpha=0.3
    )
    labels = [f"Neuron: {i}" for i, e in enumerate(vectors[:30])]
    for line in range(len(vectors[:30])):
        p1.text(tpd['tsne-2d-one'][line] + 0.01, tpd['tsne-2d-two'][line],
                labels[line], horizontalalignment='left',
                size='small', color='black', weight='light')
    # label_point(tpd["tsne-2d-one"], tpd["tsne-2d-two"], [f"Neuron: {i}" for i, e in enumerate(vectors)], plt.gca())
    plt.show()
    try_kmeans(tpd, tsne_results, vectors)


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))


def try_kmeans(tsne_output, tsne_arrays, vectors):
    for i in range(3, 11):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(tsne_arrays)
        labels = kmeans.labels_

        plt.figure(figsize=(16, 10))
        plt.title(f"{i} Clusters")
        tsne_output["labels"] = labels
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            palette=sns.color_palette("hls", i),
            hue="labels",
            data=tsne_output,
            legend="full",
            alpha=1
        )

        labels = [f"Neuron: {i}" for i, e in enumerate(vectors[:30])]
        for line in range(len(vectors[:30])):
            p1.text(tsne_output['tsne-2d-one'][line] + 0.01, tsne_output['tsne-2d-two'][line],
                    labels[line], horizontalalignment='left',
                    size='small', color='black', weight='bold')
        plt.show()

        # plt.figure(figsize=(16, 10))
        # plt.scatter(tsne_arrays[:, 0], tsne_arrays[:, 1], c=labels.astype(np.float), edgecolor="k", s=50)
        # plt.show()


def knn_clustering(tsne_output):
    nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree')
    nbrs.fit(tsne_output)



data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Predator-Static")
stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli", "Predator-Static")

stimulus_vector = get_stimulus_vector(stimulus_data, "predator 1")
all_vectors = get_all_neuron_vectors(data, "predator 1", stimulus_data, "rnn state")
all_vectors = normalise_vrvs(all_vectors)


plot_tsne_results(all_vectors, "Visual Response Vectors")
