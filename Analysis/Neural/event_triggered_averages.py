import numpy as np


from Analysis.load_data import load_data


def get_event_triggered_average(data, event_name):
    indexes = [i for i, m in enumerate(data[event_name]) if m > 0]
    neuron_averages = [0 for i in range(len(data["rnn state"][0][0]))]
    neural_data = np.squeeze(data["rnn state"])
    for i in indexes:
        for j, n in enumerate(neuron_averages):
            neuron_averages[j] += neural_data[i][j]
    for s, n in enumerate(neuron_averages):
        neuron_averages[s] = n/len(indexes)
    return neuron_averages


def get_action_triggered_average(data):
    action_triggered_averages = {str(i): [0 for i in range(len(data["rnn state"][0][0]))] for i in range(10)}
    action_counts = {str(i): 0 for i in range(10)}
    for a, n in zip(data["behavioural choice"], np.squeeze(data["rnn state"])):
        for i, nn in enumerate(n):
            action_triggered_averages[str(a)][i] += nn
        action_counts[str(a)] += 1
    for a in action_counts.keys():
        if action_counts[a] > 2:
            for i, n in enumerate(action_triggered_averages[a]):
                action_triggered_averages[a][i] = n/action_counts[a]
    return action_triggered_averages


data = load_data("even_prey_ref-4", "Naturalistic", "Naturalistic-1")
ata = get_event_triggered_average(data, "consumed")

# TODO: Create means of visualising event triggered averages: Add lines to traces for when they occur,
#  bar plots to show event triggered averages for each action for each neuron.

x = True
