import numpy as np


def normalise_within_neuron_multiple_traces(neuron_traces):
    normalised_neuron_traces = []
    for n in range(len(neuron_traces)):
        m = max([abs(min(neuron_traces[n])), abs(max(neuron_traces[n]))])
        normalised_neuron_traces.append(np.interp(neuron_traces[n], (-m, m), (-1, 1)))
    return np.array(normalised_neuron_traces)


def normalise_within_neuron(neuron_trace):
    m = max([abs(min(neuron_trace)), abs(max(neuron_trace))])
    normalised_trace = np.interp(neuron_trace, (-m, m), (-1, 1))
    return np.array(normalised_trace)


def normalise_between_neurons():
    ...



