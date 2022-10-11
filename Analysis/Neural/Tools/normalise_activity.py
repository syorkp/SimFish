import numpy as np


def normalise_within_neuron_multiple_traces(neuron_traces, zero_score_start=False):
    normalised_neuron_traces = []
    for n in range(len(neuron_traces)):
        if zero_score_start:
            neuron_traces[n] -= neuron_traces[n][0]
        m = max([abs(min(neuron_traces[n])), abs(max(neuron_traces[n]))])
        normalised_neuron = np.interp(neuron_traces[n], (-m, m), (-1, 1))

        normalised_neuron_traces.append(normalised_neuron)
    return np.array(normalised_neuron_traces)


def normalise_within_neuron(neuron_trace):
    m = max([abs(min(neuron_trace)), abs(max(neuron_trace))])
    normalised_trace = np.interp(neuron_trace, (-m, m), (-1, 1))
    return np.array(normalised_trace)


def normalise_between_neurons():
    ...



