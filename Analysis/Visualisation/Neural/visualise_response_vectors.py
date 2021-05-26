import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from Analysis.Neural.New.calculate_vrv import create_full_response_vector, create_full_stimulus_vector
from Analysis.Neural.New.label_neurons import normalise_response_vectors


def remove_initialisation_effects(vector):
    for i, neuron in enumerate(vector):
        for j, value in enumerate(neuron):
            if j % 11 == 0:
                if value == max(neuron[j: j + 11]):
                    vector[i][j] = np.mean(neuron[j: j + 11])
            elif j % 11 == 1:
                if value == max(neuron[j - 1: j + 10]):
                    vector[i][j] = np.mean(neuron[j - 1: j + 10])
    return vector


def format_func_prey(value, tick_number):
    N = int(np.round(value / 11))
    categories = ["Static-5", "Static-10", "Static-15",
                  "Left-5", "Left-10", "Left-15",
                  "Right-5", "Right-10", "Right-15",
                  "Away", "Towards"]
    if 0 <= N < 11:
        return "                  " + categories[N]
    else:
        return ""


def format_func_pred(value, tick_number):
    N = int(np.round(value / 11))
    categories = ["Static-40", "Static-60", "Static-80",
                  "Left-40", "Left-60", "Left-80",
                  "Right-40", "Right-60", "Right-80",
                  "Away", "Towards"]
    if 0 <= N < 11:
        return "                  " + categories[N]
    else:
        return ""


def format_func_both(value, tick_number):
    N = int(np.round(value / 11))
    categories = ["Static-5", "Static-10", "Static-15",
                  "Left-5", "Left-10", "Left-15",
                  "Right-5", "Right-10", "Right-15",
                  "Away", "Towards", "Static-40", "Static-60", "Static-80",
                  "Left-40", "Left-60", "Left-80",
                  "Right-40", "Right-60", "Right-80",
                  "Away", "Towards"]
    if 0 <= N < 22:
        return "                  " + categories[N]
    else:
        return ""


def pairwise_distances_sort(response_vector):
    # D = np.zeros((len(response_vector), len(response_vector)))
    # for i in range(len(response_vector)):
    #     for j in range(i, len(response_vector)):
    #         D[i,j] = np.sqrt(sum((response_vector[i,:]-response_vector[j,:])**2))
    # _ = np.lexsort(response_vector.T)
    # return response_vector[_, :]
    rv = response_vector.copy()
    r = np.sum(rv ** 2, axis=1)
    idx = np.argsort(r)
    return response_vector[idx, :]


def show_full_vector_simple_abs(response_vector, stimulus_vector, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 20)
    # idex = np.lexsort([response_vector[:, 0], response_vector[:, 11]])
    # response_vector = response_vector[idex, :]
    # response_vector = sorted(response_vector, key=lambda x: sum(x[:]))
    # response_vector = pairwise_distances_sort(response_vector)
    ax.set_title(title, fontsize=45)
    ax.pcolor(response_vector, cmap='OrRd')
    ax.set_xticks(np.linspace(0.5, len(stimulus_vector) - 0.5, len(stimulus_vector)))
    ax.set_xticklabels(stimulus_vector, rotation='vertical')
    plt.show()


def show_full_vector_simple(response_vector, stimulus_vector, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 20)
    # idex = np.lexsort([response_vector[:, 0], response_vector[:, 11]])
    # response_vector = response_vector[idex, :]
    # response_vector = sorted(response_vector, key=lambda x: sum(x[:]))
    # response_vector = pairwise_distances_sort(response_vector)
    ax.set_title(title, fontsize=45)
    ax.pcolor(response_vector, cmap='coolwarm')
    ax.set_xticks(np.linspace(0.5, len(stimulus_vector) - 0.5, len(stimulus_vector)))
    ax.set_xticklabels(stimulus_vector, rotation='vertical')
    plt.show()


def display_half_response_vector(response_vector, stimulus_vector, title, transition_points=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 20)
    ax.set_title(title, fontsize=45)
    ax.pcolor(response_vector, cmap='coolwarm')
    ax.grid(True, which='minor', axis='both', linestyle='-')#, color='k')
    ax.set_xlim(0, 121)
    ax.xaxis.set_major_locator(plt.MultipleLocator(11))
    if "Prey" in title:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_prey))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_pred))

    ax.tick_params(labelsize=15)
    ax.set_xticks(range(0, len(stimulus_vector), 11), minor=True)
    if transition_points:
        transition_points = [0] + transition_points

        # cluster_labels = [i for i in range(len(transition_points))]

        def format_func_cluster(value, tick):
            for i, tp in enumerate(transition_points):
                if value < tp:
                    return i - 1
            return len(transition_points) - 1

        ax.set_yticks(transition_points, minor=True, fontsize=20)
        ax2 = ax.secondary_yaxis("right")
        ax2.yaxis.set_major_locator(plt.FixedLocator(transition_points))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func_cluster))
        ax2.set_ylabel("Cluster")

    ax.set_xlabel("Stimulus and Position", fontsize=35)
    ax.set_ylabel("Neuron", fontsize=35)
    ax.xaxis.grid(linewidth=1, color="black")
    # ax.xaxis._axinfo["grid"]['linewidth'] = 3.
    plt.show()


def display_full_response_vector(response_vector, stimulus_vector, title, transition_points=None):
    fig, ax = plt.subplots()
    # fig.set_size_inches(18.5, 80)
    fig.set_size_inches(37, 20)
    # response_vector = sorted(response_vector, key=lambda x: sum(x[:]))
    ax.set_title(title, fontsize=45)
    ax.pcolor(response_vector, cmap='coolwarm')
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    ax.set_xlim(0, 242)
    ax.xaxis.set_major_locator(plt.MultipleLocator(11))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_both))

    ax.tick_params(labelsize=15)
    ax.set_xticks(range(0, len(stimulus_vector), 11), minor=True)
    if transition_points:
        transition_points = [0] + transition_points

        # cluster_labels = [i for i in range(len(transition_points))]

        def format_func_cluster(value, tick):
            for i, tp in enumerate(transition_points):
                if value < tp:
                    return i - 1
            return len(transition_points) - 1
        for t in transition_points:
            ax.axhline(t, color="black", linewidth=1)
        ax.set_yticks(transition_points, minor=True)
        ax2 = ax.secondary_yaxis("right")
        ax2.tick_params(axis='y', labelsize=20)
        ax2.yaxis.set_major_locator(plt.FixedLocator(transition_points))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func_cluster))
        ax2.set_ylabel("Cluster", fontsize=35)
    for t in range(0, len(response_vector[0]), 11):
        ax.axvline(t, color="black", linewidth=1)
    ax.set_xlabel("Stimulus and Position", fontsize=35)
    ax.set_ylabel("Neuron", fontsize=35)
    # ax.xaxis._axinfo["grid"]['linewidth'] = 3.
    plt.show()


def get_central_vectors(response_vector, stimulus_vector):
    new_response_vector = []
    for n, neuron in enumerate(response_vector):
        new_stimulus_vector = []
        new_neuron_vector = []
        for i, stimulus in enumerate(stimulus_vector):
            if i % 11 == 5:
                stimulus = stimulus.split("-")[:-1]
                new_name = ""
                for s in stimulus:
                    new_name = new_name + "-" + s
                new_stimulus_vector.append(new_name)
                new_neuron_vector.append(response_vector[n, i])
        new_response_vector.append(new_neuron_vector)
    return np.array(new_response_vector), new_stimulus_vector


def reduce_vector_dimensionality(response_vector, stimulus_vector):
    """
    Uses the sum of the absolute value of each point in vector to compute a responsiveness measure to each stimulus.
    """
    new_response_vector = []
    for n, neuron in enumerate(response_vector):
        new_stimulus_vector = []
        new_neuron_vector = []
        for i, stimulus in enumerate(stimulus_vector):
            if i % 11 == 5:
                stimulus = stimulus.split("-")[:-1]
                new_name = ""
                for s in stimulus:
                    new_name = new_name + "-" + s
                new_stimulus_vector.append(new_name)
                new_neuron_vector.append(sum([abs(v) for v in response_vector[n][i: i + 11]]))
        new_response_vector.append(new_neuron_vector)
    return np.array(new_response_vector), new_stimulus_vector


def get_small_size_selectivity(response_vector):
    new_response_vector = []
    for neuron in response_vector:
        if sum(neuron[0:11] + neuron[33:44] + neuron[77:88]) > 1:
            new_response_vector.append(neuron)
    return new_response_vector


def order_vectors_by_kmeans(vectors, optimal_num=None):
    # Clustering
    sil = []
    if optimal_num is None:
        for i in range(10, 50):
            kmeans = KMeans(n_clusters=i).fit(vectors)
            lab = kmeans.labels_
            sil.append(silhouette_score(vectors, lab, metric='euclidean'))
        optimal_num = sil.index(min(sil)) + 1
    kmeans = KMeans(n_clusters=optimal_num).fit(vectors)
    lab = kmeans.labels_

    # Reordering
    ordered_vectors = []
    all_clusters = set(lab)
    transition_points = []
    for cluster in all_clusters:
        for i, neuron in enumerate(vectors):
            if lab[i] == cluster:
                ordered_vectors.append(neuron)
        transition_points.append(len(ordered_vectors))
    return np.array(ordered_vectors), transition_points, lab


def order_vectors_by_agglomerative(vectors, optimal_num=None):
    # Clustering
    # dendrogram = sch.dendrogram(sch.linkage(vectors, method='ward'))
    sil = []
    if optimal_num is None:
        for i in range(10, 50):
            hc = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
            y_hc = hc.fit_predict(vectors)
            sil.append(silhouette_score(vectors, y_hc, metric='euclidean'))
        optimal_num = sil.index(min(sil)) + 1
    hc = AgglomerativeClustering(n_clusters=optimal_num, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(vectors)

    # Ordering
    ordered_vectors = []
    all_clusters = set(y_hc)
    transition_points = []
    for cluster in all_clusters:
        for i, neuron in enumerate(vectors):
            if y_hc[i] == cluster:
                ordered_vectors.append(neuron)
        transition_points.append(len(ordered_vectors))
    return np.array(ordered_vectors), transition_points[:-1], y_hc



def display_class_counts(model_names, neuron_groups, group_number):
    data = [Counter(group) for group in neuron_groups]
    tallies = [[data[j][i] for j in range(4)] for i in range(21)]
    d_tallies = pd.DataFrame({str(i): tal for i, tal in enumerate(tallies)})
    plt.boxplot(tallies)
    plt.show()
    sns.stripplot(data=d_tallies)
    plt.axhline(0, ls="--")
    plt.show()


import json

def get_transition_points_and_order(groups, vectors):
    ordered_atas = []
    indexes = [j for sub in [groups[i] for i in groups.keys()] for j in sub]
    for i in indexes:
        ordered_atas.append(vectors[i])
    transition_points = [len(groups[key]) for i, key in enumerate(groups.keys())]
    cumulative_tps = []
    for i, t in enumerate(transition_points):
        cumulative_tps.append(sum(transition_points[:i]))
    transition_points = cumulative_tps
    return transition_points, ordered_atas


with open(f"../../Categorisation-Data/latest_even.json", 'r') as f:
    data2 = json.load(f)


full_rv = create_full_response_vector("new_even_prey_ref-4", background=True)
full_rv = np.array(full_rv)
full_rv, full_rv2 = list(full_rv[:, :int(len(full_rv[0])/2)]), list(full_rv[:, int(len(full_rv[0])/2):])
full_sv = create_full_stimulus_vector("new_even_prey_ref-4")
full_rv = normalise_response_vectors(full_rv)

tps, full_rv = get_transition_points_and_order(data2["new_even_prey_ref-4"], full_rv)

display_full_response_vector(full_rv, full_sv, "All Stimuli", tps)


# for i in range(3, 7):
#     full_rv = create_full_response_vector(f"new_differential_prey_ref-{str(i)}", background=True)
#     full_rv = np.array(full_rv)
#     full_rv, full_rv2 = list(full_rv[:, :int(len(full_rv[0])/2)]), list(full_rv[:, int(len(full_rv[0])/2):])
#     full_sv = create_full_stimulus_vector(f"new_differential_prey_ref-{str(i)}")
#     full_rv = normalise_response_vectors(full_rv)
#     # full_rv2 = normalise_response_vectors(full_rv2)
#     full_rv, transition_points, neuron_labels = order_vectors_by_kmeans(full_rv)
#     print(len(transition_points))
#
# for i in [4, 5, 6, 8]:
#     full_rv = create_full_response_vector(f"new_even_prey_ref-{str(i)}", background=True)
#     full_rv = np.array(full_rv)
#     full_rv, full_rv2 = list(full_rv[:, :int(len(full_rv[0])/2)]), list(full_rv[:, int(len(full_rv[0])/2):])
#     full_sv = create_full_stimulus_vector(f"new_even_prey_ref-{str(i)}")
#     full_rv = normalise_response_vectors(full_rv)
#     full_rv, transition_points, neuron_labels = order_vectors_by_kmeans(full_rv)
#     print(len(transition_points))
#     display_full_response_vector(full_rv, full_sv, "All Stimuli", transition_points)

# Single model:
# full_rv = create_full_response_vector("even_prey_ref-5")
# full_rv = normalise_response_vectors(full_rv)
# full_sv = create_full_stimulus_vector("even_prey_ref-5")
#
# full_rv, transition_points, neuron_labels = order_vectors_by_kmeans(full_rv, 21)
# display_full_response_vector(full_rv, full_sv, "Prey Stimuli", transition_points)
#
#
# full_rv = create_full_response_vector("new_differential_prey_ref-4")
# full_rv = normalise_response_vectors(full_rv)
# full_sv = create_full_stimulus_vector("new_differential_prey_ref-4")
#
# full_rv, transition_points, neuron_labels = order_vectors_by_kmeans(full_rv, 21)
# display_full_response_vector(full_rv, full_sv, "Prey Stimuli", transition_points)


# full_rv = remove_initialisation_effects(full_rv)
# full_rv = order_vectors_by_kmeans(full_rv)
# display_full_response_vector(full_rv, full_sv, "Full")
# prey_rv, transition_points, neuron_labels = order_vectors_by_agglomerative(full_rv[:, :121], 21)
# display_full_response_vector(prey_rv, prey_sv, "Prey Stimuli", transition_points)

# prey_rv, transition_points, neuron_labels = order_vectors_by_kmeans(full_rv[:, :121], 21)
# display_full_response_vector(prey_rv, prey_sv, "Prey Stimuli", transition_points)

# prey_rv, transition_points, neuron_labels = order_vectors_by_agglomerative(full_rv[:, :121])
# prey_rv = get_small_size_selectivity(prey_rv)
# pred_rv = order_vectors_by_kmeans(full_rv[:, 121:])
# pred_sv = full_sv[121:]
#
# simple_rv, simple_sv = get_central_vectors(full_rv, full_sv)
# simple_rv, simple_sv = reduce_vector_dimensionality(full_rv, full_sv)
# simple_rv = order_vectors_by_kmeans(simple_rv)
# display_full_response_vector(full_rv, full_sv, "Full")
# display_full_response_vector(prey_rv, prey_sv, "Prey Stimuli", transition_points)
# display_full_response_vector(pred_rv, pred_sv, "Predator Stimuli")
# show_full_vector_simple_abs(simple_rv, simple_sv, "Simplified")
