import numpy as np


from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data


def get_stimulus_vector(stimulus_data, stimulus):
    """Returns a list of all the stimulus features presented"""
    angles = []

    for period in stimulus_data:
        angles.append(period[stimulus]["Angle"])

    return angles


def get_all_neuron_vectors(all_data, stimulus, stimulus_data, neuron_type):
    # TODO: note, this probably wont be appropriate for conv layers as the data dimensions will be different.
    n_neurons = all_data[neuron_type].shape[2]
    vectors = []
    for i in range(n_neurons):
        neural_data = all_data[neuron_type][:,:,i]
        vector = get_neuron_vector(neural_data, stimulus_data, stimulus)
        vectors.append(vector)
    return vectors


def get_neuron_vector(neural_data, stimulus_data, stimulus):
    vector = []
    for period in stimulus_data:
        interval = period[stimulus]["Onset"] - period[stimulus]["Pre-onset"]
        vector.append(calculate_scalar_value(neural_data, period[stimulus]["Pre-onset"], period[stimulus]["Onset"], period[stimulus]["Onset"] + interval))
    return vector


def calculate_scalar_value(neural_data, t_start, t_mid, t_fin):
    baseline = np.mean(neural_data[t_start: t_mid])
    response = np.mean(neural_data[t_mid: t_fin])
    return (response - baseline)/baseline


# data = load_data("changed_penalties-1", "Controlled_Visual_Stimuli", "Curved_prey")
# stimulus_data = load_stimulus_data("changed_penalties-1", "Controlled_Visual_Stimuli")
#
# stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
# all_vectors = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn state")


x = True


