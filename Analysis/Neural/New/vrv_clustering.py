import json

from sklearn.cluster import KMeans

from Analysis.Neural.New.calculate_vrv import create_full_response_vector, create_full_stimulus_vector
from Analysis.Neural.New.label_neurons import normalise_response_vectors
from Analysis.Visualisation.Neural.visualise_response_vectors import display_full_response_vector


def knn_clustering_assign_categories(response_vectors, stimulus_vector, optimal_num):
    all_vectors = []

    for vector in response_vectors:
        all_vectors += vector
    all_vectors = normalise_response_vectors(all_vectors)

    kmeans = KMeans(n_clusters=optimal_num, n_init=20).fit(all_vectors)
    lab = kmeans.labels_
    model_labels = []
    for i in range(0, len(all_vectors), len(response_vectors[0])):
        labels = lab[i:i + len(response_vectors[0])]
        model_labels.append(labels)

    # Reordering
    ordered_vectors = []
    all_clusters = set(lab)
    transition_points = []
    for cluster in all_clusters:
        for i, neuron in enumerate(all_vectors):
            if lab[i] == cluster:
                ordered_vectors.append(neuron)
        transition_points.append(len(ordered_vectors))
    display_full_response_vector(ordered_vectors, stimulus_vector, "All Stimuli", transition_points)

    return model_labels, transition_points


def save_neuron_groups(model_names, neuron_groups, group_number, group_name):
    data = {model: {str(i): [] for i in range(group_number)} for model in model_names}
    for i, model in enumerate(model_names):
        for j, neuron in enumerate(neuron_groups[i]):
            data[model][str(neuron)].append(j)
    with open(f"../../Categorisation-Data/{group_name}.json", 'w') as f:
        json.dump(data, f, indent=4)





# Full vector
full_rv1 = create_full_response_vector("new_even_prey_ref-4", background=False)
full_rv2 = create_full_response_vector("new_even_prey_ref-5", background=False)
full_rv3 = create_full_response_vector("new_even_prey_ref-6", background=False)
full_rv4 = create_full_response_vector("new_even_prey_ref-8", background=False)

full_sv = create_full_stimulus_vector("new_even_prey_ref-4")

model_l, transition_points = knn_clustering_assign_categories([full_rv1, full_rv2, full_rv3, full_rv4], full_sv, 30)
save_neuron_groups(["new_even_prey_ref-4",
                    "new_even_prey_ref-5",
                    "new_even_prey_ref-6",
                    "new_even_prey_ref-8"], model_l, 30, "final_even")


# Do clustering over many models:
# full_rv1 = create_full_response_vector("new_differential_prey_ref-3")
# full_rv2 = create_full_response_vector("new_differential_prey_ref-4")
# full_rv3 = create_full_response_vector("new_differential_prey_ref-5")
# full_rv4 = create_full_response_vector("new_differential_prey_ref-6")
#
# full_sv = create_full_stimulus_vector("new_differential_prey_ref-4")
#
# model_l = knn_clustering_assign_categories([full_rv1, full_rv2, full_rv3, full_rv4], full_sv, 35)
# save_neuron_groups(["new_differential_prey_ref-3", "new_differential_prey_ref-4", "new_differential_prey_ref-5",
#                     "new_differential_prey_ref-6"], model_l, 35)
# display_class_counts(["even_prey_ref-5", "even_prey_ref-6", "even_prey_ref-7", "even_prey_ref-4",], model_l, 21)
#
# many_rv = normalise_response_vectors(many_rv)
# many_rv, transition_points, neuron_labels = order_vectors_by_kmeans(many_rv, 30)
# display_full_response_vector(many_rv, full_sv, "Prey Stimuli", transition_points)
