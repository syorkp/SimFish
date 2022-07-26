import numpy as np


def normalise_within_neuron(neuron_traces):
    normalised_neuron_traces = []
    for n in range(len(neuron_traces)):
        m = max([abs(min(neuron_traces[n])), abs(max(neuron_traces[n]))])
        normalised_neuron_traces.append(np.interp(neuron_traces[n], (-m, m), (-1, 1)))
    return np.array(normalised_neuron_traces)


def normalise_between_neurons():
    ...



