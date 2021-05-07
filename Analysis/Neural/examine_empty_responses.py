import numpy as np

from Analysis.load_data import load_data


def get_free_swimming_indexes(data):
    """Requires the following data: position, prey_positions, predator. Assumes square arena 1500."""
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    wall_timestamps = [i for i, p in enumerate(data["position"]) if 200<p[0]<1300 and 200<p[1]<1300]
    prey_timestamps = []
    sensing_distance = 200
    for i, p in enumerate(data["position"]):
        for prey in data["prey_positions"][i]:
            sensing_area = [[p[0] - sensing_distance,
                             p[0] + sensing_distance],
                            [p[1] - sensing_distance,
                             p[1] + sensing_distance]]
            near_prey = sensing_area[0][0] <= prey[0] <= sensing_area[0][1] and \
                         sensing_area[1][0] <= prey[1] <= sensing_area[1][1]
            if near_prey:
                prey_timestamps.append(i)
                break
    # Check prey near at each step and add to timestamps.
    null_timestamps = predator_timestamps + wall_timestamps + prey_timestamps
    null_timestamps = set(null_timestamps)
    desired_timestamps = [i for i in range(len(data["behavioural choice"])) if i not in null_timestamps]
    return desired_timestamps


def get_space_triggered_averages(data, indexes):
    neuron_baseline = [np.mean(data["rnn state"][:, :, i]) for i in range(len(data["rnn state"][0][0]))]
    neuron_averages = [0 for i in range(len(data["rnn state"][0][0]))]
    for i in indexes:
        for n in range(len(data["rnn state"][0][0])):
            neuron_averages[n] += data["rnn state"][i, 0, n]
    for n, neuron in enumerate(neuron_averages):
        neuron_averages[n] = (((neuron/len(indexes))-neuron_baseline[n])/neuron_baseline[n]) * 100
    return neuron_averages


d = load_data("even_prey_ref-5", "Empty-Environment", "Empty-1")
ind = get_free_swimming_indexes(d)
sta = get_space_triggered_averages(d, ind)
x = True