import matplotlib.pyplot as plt
import numpy
from matplotlib import colors
import numpy as np

from Analysis.Neural.calculate_vrv import create_full_response_vector, create_full_stimulus_vector
from Analysis.Neural.label_neurons import normalise_response_vectors


def remove_initialisation_effects(vector):
    for i, neuron in enumerate(vector):
        for j, value in enumerate(neuron):
            if j%11 == 0:
                if value == max(neuron[j: j+11]):
                    vector[i][j] = np.mean(neuron[j: j+11])
            elif j%11 == 1:
                if value == max(neuron[j-1: j+10]):
                    vector[i][j] = np.mean(neuron[j-1: j+10])
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

def pairwise_distances_sort(response_vector):
    # D = np.zeros((len(response_vector), len(response_vector)))
    # for i in range(len(response_vector)):
    #     for j in range(i, len(response_vector)):
    #         D[i,j] = np.sqrt(sum((response_vector[i,:]-response_vector[j,:])**2))
    # _ = np.lexsort(response_vector.T)
    # return response_vector[_, :]
    rv = response_vector.copy()
    r = np.sum(rv**2, axis=1)
    idx = np.argsort(r)
    return response_vector[idx, :]


def show_full_vector_simple(response_vector, stimulus_vector, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 20)
    # idex = np.lexsort([response_vector[:, 0], response_vector[:, 11]])
    # response_vector = response_vector[idex, :]
    #response_vector = sorted(response_vector, key=lambda x: sum(x[:]))
    # response_vector = pairwise_distances_sort(response_vector)
    ax.set_title(title, fontsize=45)
    ax.pcolor(response_vector, cmap='coolwarm')
    ax.set_xticks(np.linspace(0.5, len(stimulus_vector)-0.5, len(stimulus_vector)))
    ax.set_xticklabels(stimulus_vector, rotation='vertical')
    plt.show()


def display_full_response_vector(response_vector, stimulus_vector, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 20)
    # response_vector = sorted(response_vector, key=lambda x: sum(x[:]))
    ax.set_title(title, fontsize=45)
    ax.pcolor(response_vector, cmap='coolwarm')
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    ax.set_xlim(0, 121)
    ax.xaxis.set_major_locator(plt.MultipleLocator(11))
    if "Prey" in title:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_prey))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_pred))

    ax.tick_params(labelsize=15)
    ax.set_xticks(range(0, len(stimulus_vector), 11), minor=True)
    ax.set_xlabel("Stimulus and Position", fontsize=35)
    ax.set_ylabel("Neuron", fontsize=35)
    ax.xaxis.grid(linewidth=1, color="black")
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


from sklearn.cluster import KMeans

def order_vectors_by_kmeans(vectors):
    ordered_vectors = []
    kmeans = KMeans(n_clusters=30).fit(vectors)
    lab = kmeans.labels_
    for cluster in set(lab):
        for i, neuron in enumerate(vectors):
            if lab[i] == cluster:
                ordered_vectors.append(neuron)
    return np.array(ordered_vectors)

#
# full_rv = create_full_response_vector("even_prey_ref-5")
# full_rv = normalise_response_vectors(full_rv)
# full_sv = create_full_stimulus_vector("even_prey_ref-5")
# full_rv = remove_initialisation_effects(full_rv)
#
# prey_rv = order_vectors_by_kmeans(full_rv[:, :121])
# prey_sv = full_sv[:121]
# pred_rv = order_vectors_by_kmeans(full_rv[:, 121:])
# pred_sv = full_sv[121:]
#
# simple_rv, simple_sv = get_central_vectors(full_rv, full_sv)
# simple_rv = order_vectors_by_kmeans(simple_rv)
# # display_full_response_vector(full_rv, full_sv)
# display_full_response_vector(prey_rv, prey_sv, "Prey Stimuli")
# display_full_response_vector(pred_rv, pred_sv, "Predator Stimuli")
# show_full_vector_simple(simple_rv, simple_sv, "Simplified")