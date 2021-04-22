import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data
from Analysis.Neural.calculate_vrv import get_all_neuron_vectors, get_stimulus_vector, normalise_vrvs, create_full_response_vector
from Analysis.Neural.label_neurons import create_full_stimulus_vector, label_all_units_selectivities, assign_neuron_names


def plot_tsne_results(vectors, title="None", labels=None):
    for p in range(5, 10, 5):
        tsne = TSNE(n_components=2, n_iter=1000, perplexity=p)
        tsne_results = tsne.fit_transform(vectors)

        tpd = {}

        tpd['tsne-2d-one'] = tsne_results[:, 0]
        tpd['tsne-2d-two'] = tsne_results[:, 1]
        tpd['Point'] = labels
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
        # labels = [f"Neuron: {i}" for i, e in enumerate(vectors[:30])]
        # for line in range(len(vectors[:30])):
        #     p1.text(tpd['tsne-2d-one'][line] + 0.01, tpd['tsne-2d-two'][line],
        #             labels[line], horizontalalignment='left',
        #             size='small', color='black', weight='light')
        # label_point(tpd["tsne-2d-one"], tpd["tsne-2d-two"], [f"Neuron: {i}" for i, e in enumerate(vectors)], plt.gca())
        plt.show()
        # try_kmeans(tpd, tsne_results, vectors, p)


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))


def try_kmeans(tsne_output, tsne_arrays, vectors, p):
    for i in range(4, 5):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(tsne_arrays)
        labels = kmeans.labels_

        # plt.figure(figsize=(16, 10))
        plt.figure(figsize=(50, 30))
        plt.title(f"{i} Clusters, p={p}")
        tsne_output["labels"] = labels
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            palette=sns.color_palette("hls", i),
            hue="labels",
            data=tsne_output,
            legend="full",
            alpha=1
        )

        labels = [f"Neuron: {i}" for i, e in enumerate(vectors)]
        for line in range(len(vectors)):
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

#
# # Good results:
# data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Predator-Static")
# stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli", "Predator-Static")
#
# stimulus_vector = get_stimulus_vector(stimulus_data, "predator 1")
# all_vectors = get_all_neuron_vectors(data, "predator 1", stimulus_data, "rnn state")
# # all_vectors = normalise_vrvs(all_vectors)
#
# plot_tsne_results(all_vectors, "Visual Response Vectors")
#
#
# # Concatenation of response vectors
# data1 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Predator-Static")
# data2 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Predator-Moving-Left")
# data3 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Predator-Moving-Right")
# data4 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Static")
# data5 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-Left")
# data6 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-Right")
# data7 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-1")
# data8 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-2")
# data9 = load_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-3")
#
# stimulus_data1 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Predator-Static")
# stimulus_data2 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Predator-Moving-Left")
# stimulus_data3 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Predator-Moving-Right")
# stimulus_data4 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Static")
# stimulus_data5 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-Left")
# stimulus_data6 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-Right")
# stimulus_data7 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-1")
# stimulus_data8 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-2")
# stimulus_data9 = load_stimulus_data("even_prey_ref-4", "Controlled_Visual_Stimuli", "Prey-Moving-3")
#
# stimulus_vector1 = get_stimulus_vector(stimulus_data1, "predator 1")
# stimulus_vector2 = get_stimulus_vector(stimulus_data2, "predator 1")
# stimulus_vector3 = get_stimulus_vector(stimulus_data3, "predator 1")
# stimulus_vector4 = get_stimulus_vector(stimulus_data4, "prey 1")
# stimulus_vector5 = get_stimulus_vector(stimulus_data5, "prey 1")
# stimulus_vector6 = get_stimulus_vector(stimulus_data6, "prey 1")
# stimulus_vector7 = get_stimulus_vector(stimulus_data7, "prey 1")
# stimulus_vector8 = get_stimulus_vector(stimulus_data8, "prey 1")
# stimulus_vector9 = get_stimulus_vector(stimulus_data9, "prey 1")
#
# vector1 = get_all_neuron_vectors(data1, "predator 1", stimulus_data1, "rnn state")
# vector2 = get_all_neuron_vectors(data2, "predator 1", stimulus_data2, "rnn state")
# vector3 = get_all_neuron_vectors(data3, "predator 1", stimulus_data3, "rnn state")
# vector4 = get_all_neuron_vectors(data4, "prey 1", stimulus_data4, "rnn state")
# vector5 = get_all_neuron_vectors(data5, "prey 1", stimulus_data5, "rnn state")
# vector6 = get_all_neuron_vectors(data6, "prey 1", stimulus_data6, "rnn state")
# vector7 = get_all_neuron_vectors(data7, "prey 1", stimulus_data7, "rnn state")
# vector8 = get_all_neuron_vectors(data8, "prey 1", stimulus_data8, "rnn state")
# vector9 = get_all_neuron_vectors(data9, "prey 1", stimulus_data9, "rnn state")
#
# all_vectors = np.concatenate([vector1, vector2, vector3, vector4, vector5, vector6, vector7, vector8, vector9], axis=1).tolist()
# all_vectors = np.concatenate([vector2, vector3], axis=1).tolist()
# normalised_vectors = normalise_vrvs(all_vectors)

full_rv = create_full_response_vector("even_prey_ref-7")
full_sv = create_full_stimulus_vector(f"even_prey_ref-7")
sel = label_all_units_selectivities(full_rv, full_sv)
cat = assign_neuron_names(sel)
plot_tsne_results(full_rv, "Visual Response Vectors", cat)
full_rv = np.array(full_rv)
# plot_tsne_results(normalised_vectors, "Visual Response Vectors")
