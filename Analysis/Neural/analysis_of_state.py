import numpy as np

from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data

from Analysis.Visualisation.display_many_neurons import plot_traces


def get_stimulus_periods():
    ...


def normalise_activity(data, average):

    return data


def normalise_all_traces(all_data):
    all_new_traces = []
    n_neurons = all_data["rnn state"].shape[2]
    for i in range(n_neurons):
        neural_data = all_data["rnn state"][:, :, i]
        average_activity = np.mean(neural_data)
        for i, point in enumerate(neural_data):
            neural_data[i] = point - average_activity
        all_new_traces.append(neural_data)
    return all_new_traces


def remove_arbitrary_traces(normalised_data):
    to_remove = []
    for n, neuron in enumerate(normalised_data):
        increasing_values = [value for i, value in enumerate(neuron) if value >= neuron[i-1]]
        decreasing_values = [value for i, value in enumerate(neuron) if value <= neuron[i-1]]
        if len(increasing_values) > len(neuron) - 30 or len(decreasing_values) > len(neuron) - 30:
            to_remove.append(n)
    for index in sorted(to_remove, reverse=True):
        del normalised_data[index]
    return normalised_data


def get_deviation_periods(normalised_trace):
    timesteps_above = []
    timesteps_below = []
    for i, n in enumerate(normalised_trace):
        if n > 0:
            timesteps_below.append(i)
        else:
            timesteps_above.append(i)
    return timesteps_above, timesteps_below


def deviation_durations(timesteps):
    continuous_trace = 1
    for i, step in enumerate(timesteps):
        if i == 0:
            pass
        else:
            if step - 1 == timesteps[i-1]:
                continuous_trace += 1
            else:
                if continuous_trace != 1:
                    print(continuous_trace)
                continuous_trace = 1



data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Curved_prey")
stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli")

normalised_traces = normalise_all_traces(data)
normalised_traces = remove_arbitrary_traces(normalised_traces)

example_n = normalised_traces[332]
a, b = get_deviation_periods(example_n)
deviation_durations(a)
print("---")
deviation_durations(b)

# normalised_traces = normalised_traces[:50]
# plot_multiple_traces(normalised_traces)

#
# unit_activity = [[data["rnn state"][i-1][0][j] for i in data["step"]] for j in range(50)]
# plot_multiple_traces(unit_activity)

x = True