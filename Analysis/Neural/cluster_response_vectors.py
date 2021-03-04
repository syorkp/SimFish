import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data
from Analysis.Neural.calculate_vrv import get_all_neuron_vectors, get_stimulus_vector


def plot_tsne_results(vectors, order=2, title="None"):
    while order > 1:
        transition_probabilities = np.vstack(vectors)
        order -= 1

    tsne = TSNE(n_components=2)
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
        data=tpd,
        legend="full",
        alpha=0.3
    )
    labels = [f"Neuron: {i}" for i, e in enumerate(vectors)]
    for line in range(len(vectors)):
        p1.text(tpd['tsne-2d-one'][line] + 0.01, tpd['tsne-2d-two'][line],
                labels[line], horizontalalignment='left',
                size='small', color='black', weight='light')
    # label_point(tpd["tsne-2d-one"], tpd["tsne-2d-two"], [f"Neuron: {i}" for i, e in enumerate(vectors)], plt.gca())
    plt.show()


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))


def try_kmeans(response_vectors):
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(all_vectors)
        print(kmeans.labels_)


data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Curved_prey")
stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli")

stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
all_vectors = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn state")

plot_tsne_results(all_vectors, 5, "Visual Response Vectors")
