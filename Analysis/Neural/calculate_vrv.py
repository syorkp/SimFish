import numpy as np
import math

from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data


def get_stimulus_vector(stimulus_data, stimulus):
    """Returns a list of all the stimulus features presented"""
    angles = []
    for period in stimulus_data:
        angles.append(period[stimulus]["Angle"])

    return angles


def get_all_neuron_vectors(all_data, stimulus, stimulus_data, neuron_type):
    n_neurons = all_data[neuron_type].shape[2]
    vectors = []
    for i in range(n_neurons):
        neural_data = all_data[neuron_type][:, :, i]
        vector = get_neuron_vector(neural_data, stimulus_data, stimulus)
        vectors.append(vector)
    return vectors


def get_conv_neuron_vectors(all_data, stimulus, stimulus_data, neuron_type):
    n_neurons = all_data[neuron_type].shape[-1]
    vectors = []
    for i in range(n_neurons):
        neural_data = all_data[neuron_type][:, :, :, i]
        vector = get_conv_neuron_vector(neural_data, stimulus_data, stimulus)
        vectors.append(vector)
    return vectors


def get_neuron_vector(neural_data, stimulus_data, stimulus):
    vector = []
    for period in stimulus_data:
        interval = period[stimulus]["Onset"] - period[stimulus]["Pre-onset"]
        vector.append(calculate_scalar_value(neural_data, period[stimulus]["Pre-onset"], period[stimulus]["Onset"],
                                             period[stimulus]["Onset"] + interval))
    return vector


def get_conv_neuron_vector(neural_data, stimulus_data, stimulus):
    conv_vectors = [[] for _ in range(4)]
    for channel in range(4):
        for period in stimulus_data:
            interval = period[stimulus]["Onset"] - period[stimulus]["Pre-onset"]
            conv_vectors[channel].append(
                calculate_scalar_value(neural_data, period[stimulus]["Pre-onset"], period[stimulus]["Onset"],
                                       period[stimulus]["Onset"] + interval))
    return conv_vectors


def calculate_scalar_value(neural_data, t_start, t_mid, t_fin):
    baseline = np.mean(neural_data[t_start: t_mid])
    response = np.mean(neural_data[t_mid: t_fin])
    if math.isnan((response - baseline) / baseline):
        return 0
    else:
        return (response - baseline) / baseline


def normalise_vrvs(vectors):
    vectors = np.array(vectors)
    for i, v in enumerate(vectors[0]):
        vectors[:, i] = np.interp(vectors[:, i], (-100, 100), (-1, 1))
    return vectors


def create_full_stimulus_vector(model_name, background=False):
    full_stimulus_vector = []
    if background:
        file_precursors = ["Prey", "Predator", "Background-Prey", "Background-Predator"]
    else:
        file_precursors = ["Prey", "Predator"]
    prey_assay_ids = ["Prey-Static-5", "Prey-Static-10", "Prey-Static-15",
                      "Prey-Left-5", "Prey-Left-10", "Prey-Left-15",
                      "Prey-Right-5", "Prey-Right-10", "Prey-Right-15",
                      "Prey-Away", "Prey-Towards"]
    predator_assay_ids = ["Predator-Static-40", "Predator-Static-60", "Predator-Static-80",
                          "Predator-Left-40", "Predator-Left-60", "Predator-Left-80",
                          "Predator-Right-40", "Predator-Right-60", "Predator-Right-80",
                          "Predator-Away", "Predator-Towards"]
    for file_p in file_precursors:
        if "Prey" in file_p:
            for aid in prey_assay_ids:
                stimulus_data = load_stimulus_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
                stimulus_vector = [aid + "-" + str(s) for s in stimulus_vector]
                full_stimulus_vector += stimulus_vector

        elif "Predator" in file_p:
            for aid in predator_assay_ids:
                stimulus_data = load_stimulus_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                stimulus_vector = get_stimulus_vector(stimulus_data, "predator 1")
                stimulus_vector = [aid + "-" + str(s) for s in stimulus_vector]
                full_stimulus_vector += stimulus_vector

    return full_stimulus_vector


def create_full_response_vector(model_name, background=False):
    # Creates the full 484 dimensional response vector.
    response_vectors = [[] for i in range(512)]
    file_precursors = ["Prey", "Predator", "Background-Prey", "Background-Predator"]
    if background:
        file_precursors = ["Prey", "Predator", "Background-Prey", "Background-Predator"]
    else:
        file_precursors = ["Prey", "Predator"]
    prey_assay_ids = ["Prey-Static-5", "Prey-Static-10", "Prey-Static-15",
                      "Prey-Left-5", "Prey-Left-10", "Prey-Left-15",
                      "Prey-Right-5", "Prey-Right-10", "Prey-Right-15",
                      "Prey-Away", "Prey-Towards"]
    predator_assay_ids = ["Predator-Static-40", "Predator-Static-60", "Predator-Static-80",
                          "Predator-Left-40", "Predator-Left-60", "Predator-Left-80",
                          "Predator-Right-40", "Predator-Right-60", "Predator-Right-80",
                          "Predator-Away", "Predator-Towards"]
    for file_p in file_precursors:
        if "Prey" in file_p:
            for aid in prey_assay_ids:
                data = load_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                stimulus_data = load_stimulus_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                new_vector_section = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn state")
                for i, n in enumerate(response_vectors):
                    response_vectors[i] = n + new_vector_section[i]
        elif "Predator" in file_p:
            for aid in predator_assay_ids:
                data = load_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                stimulus_data = load_stimulus_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                new_vector_section = get_all_neuron_vectors(data, "predator 1", stimulus_data, "rnn state")
                for i, n in enumerate(response_vectors):
                    response_vectors[i] = n + new_vector_section[i]
    return response_vectors


# full_rv = create_full_response_vector("even_prey_ref-5")
# x = True


# data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Curved_prey")
# stimulus_data = load_stimulus_data("changed_penalties-1", "Controlled_Visual_Stimuli")

# stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
# # all_vectors = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn_state")
# conv_vectors = get_conv_neuron_vectors(data, "prey 1", stimulus_data, "left_conv_4")


# x = True
