import copy

import numpy as np
import matplotlib.pyplot as plt

from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.load_data import load_data
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces


def get_rnn_interconnectivity(all_network_variables):
    keys = all_network_variables.keys()
    relevant_keys = [key for key in keys if "lstm_cell" in key and "bias" not in key]
    rnn_related = {key: all_network_variables[key] for key in relevant_keys}
    # Weights are in this format along axis 1: w_i, w_C, w_f, w_o
    # And along axis 0: x, h

    # input_size = 512
    # targets = rnn_related[list(rnn_related.keys())[0]]
    # w_i, w_C, w_f, w_o = np.split(targets, 4, axis=1)
    # w_xi = w_i[:input_size, :]
    # w_hi = w_i[input_size:, :]
    #
    # w_xC = w_C[:input_size, :]
    # w_hC = w_C[input_size:, :]
    #
    # w_xf = w_f[:input_size, :]
    # w_hf = w_f[input_size:, :]
    #
    # w_xo = w_o[:input_size, :]
    # w_ho = w_o[input_size:, :]

    all_forget_gate_weights = {key: rnn_related[key][:, 1024:1536] for key in rnn_related.keys()}
    return all_forget_gate_weights


def plot_hist_all_weights(all_weights):
    all_weights = all_weights.flatten()
    plt.hist(all_weights, bins=100)
    plt.savefig("hist_all_weights.png")
    plt.show()


def get_strongly_connected_neuron_pairs(all_weights, threshold, absolute_weights):
    """Returns i, j indices of neurons for which their reciprocal connections are both super threshold."""
    all_weights = all_weights[:512]

    xp, yp = np.arange(512), np.arange(512)
    xy, py = np.meshgrid(xp, yp)
    xy = np.expand_dims(xy, 2)
    py = np.expand_dims(py, 2)
    all_pairs = np.concatenate((xy, py), axis=2).reshape(-1, 2)
    all_pairs = np.array([pair for pair in all_pairs if pair[0] != pair[1]])

    if absolute_weights:
        inter_differences = (np.absolute(all_weights[all_pairs[:, 0], all_pairs[:, 1]]) > threshold) * \
                            (np.absolute(all_weights[all_pairs[:, 1], all_pairs[:, 0]]) > threshold)
    else:
        inter_differences = (all_weights[all_pairs[:, 0], all_pairs[:, 1]] > threshold) * \
                            (all_weights[all_pairs[:, 1], all_pairs[:, 0]] > threshold)
    all_pairs = all_pairs[inter_differences]
    all_pairs = np.sort(all_pairs, axis=1)
    all_pairs = np.unique(all_pairs, axis=0)
    return all_pairs


def get_weights_to_self(all_weights):
    indices = np.expand_dims(np.arange(0, 512), 1)
    indices = np.concatenate((indices, indices), axis=1)
    weights_to_self = all_weights[indices[:, 0], indices[:, 1]]
    plt.hist(weights_to_self, bins=10)
    plt.savefig("hist_weights_to_self.png")
    plt.show()


def compare_weights_to_and_from(all_weights):
    """Looks for a correlation in the weights between individual neurons."""
    all_weights = all_weights[:512]

    xp, yp = np.arange(512), np.arange(512)
    xy, py = np.meshgrid(xp, yp)
    xy = np.expand_dims(xy, 2)
    py = np.expand_dims(py, 2)
    all_pairs = np.concatenate((xy, py), axis=2).reshape(-1, 2)
    all_pairs = np.array([pair for pair in all_pairs if pair[0] != pair[1]])

    inter_differences = all_weights[all_pairs[:, 0], all_pairs[:, 1]] - all_weights[all_pairs[:, 1], all_pairs[:, 0]]
    squared_inter_differences = inter_differences ** 2

    np.random.shuffle(all_weights)
    inter_differences_random = all_weights[all_pairs[:, 0], all_pairs[:, 1]] - all_weights[all_pairs[:, 1], all_pairs[:, 0]]
    squared_inter_differences_random = inter_differences_random ** 2

    plt.hist(inter_differences, bins=100, alpha=0.5)
    plt.hist(inter_differences_random, bins=100, alpha=0.5)
    plt.savefig("hist_difference_random_connections.png")
    plt.clf()

    print(f"""Mean squared error - inter neuron connections: {np.mean(squared_inter_differences)}
Mean squared error - random neuron connections: {np.mean(squared_inter_differences_random)}
""")


def get_activity_profiles_by_indices_pairs(data, indices_pairs):
    activity = [[data["rnn_state_actor"][:, 0, 0, indices[0]], data["rnn_state_actor"][:, 0, 0, indices[1]]]
                for indices in indices_pairs]
    return np.array(activity)


def plot_rnn_activity_pairs(activity_by_pair):
    num_traces = activity_by_pair.shape[0]
    fig, axs = plt.subplots(num_traces, figsize=(10, 20))
    for trace in range(num_traces):
        axs[trace].plot(activity_pairs[trace, 0])
        axs[trace].plot(activity_pairs[trace, 1])
    plt.savefig("Comparison plots.png")


def compute_paired_similarity_metrics(activity_profiles):
    # Normalise traces
    activity_profiles[:, 0, :] = normalise_within_neuron_multiple_traces(activity_profiles[:, 0, :])
    activity_profiles[:, 1, :] = normalise_within_neuron_multiple_traces(activity_profiles[:, 1, :])

    differentiated_activity_profiles = activity_profiles[:, :, 1:] - activity_profiles[:, :, :-1]

    squared_difference = (activity_profiles[:, 0, :] - activity_profiles[:, 1, :]) ** 2
    squared_difference_diff = (differentiated_activity_profiles[:, 0, :] - differentiated_activity_profiles[:, 1, :]) ** 2

    mean_squared_difference = np.mean(squared_difference, axis=1)
    mean_squared_difference_diff = np.mean(squared_difference_diff, axis=1)

    return np.mean(mean_squared_difference), np.mean(mean_squared_difference_diff)


def group_neuron_pairs_by_connection_strength(all_weights, bin_size, absolute_weights):
    """Note: returns pairs for which reciprocal connections fall into shared bins, naturally will exclude most possible
    pairs"""

    max_weight, min_weight = np.max(all_weights), np.min(all_weights)
    diff = max_weight - min_weight
    bins = np.linspace(min_weight, max_weight, int(diff/bin_size))

    xp, yp = np.arange(512), np.arange(512)
    xy, py = np.meshgrid(xp, yp)
    xy = np.expand_dims(xy, 2)
    py = np.expand_dims(py, 2)
    all_pairs = np.concatenate((xy, py), axis=2).reshape(-1, 2)
    all_pairs = np.array([pair for pair in all_pairs if pair[0] != pair[1]])

    binned_pairs = []
    for i, bin in enumerate(bins[:-1]):
        if absolute_weights:
            inter_differences = (np.absolute(all_weights[all_pairs[:, 0], all_pairs[:, 1]]) > bin) * \
                                (np.absolute(all_weights[all_pairs[:, 0], all_pairs[:, 1]]) <= bins[i+1]) * \
                                (np.absolute(all_weights[all_pairs[:, 1], all_pairs[:, 0]]) > bin) * \
                                (np.absolute(all_weights[all_pairs[:, 1], all_pairs[:, 0]]) <= bins[i+1])
        else:
            inter_differences = (all_weights[all_pairs[:, 0], all_pairs[:, 1]] > bin) * \
                                (all_weights[all_pairs[:, 0], all_pairs[:, 1]] <= bins[i+1]) * \
                                (all_weights[all_pairs[:, 1], all_pairs[:, 0]] > bin) * \
                                (all_weights[all_pairs[:, 1], all_pairs[:, 0]] <= bins[i+1])
        selected_pairs = all_pairs[inter_differences]
        selected_pairs = np.sort(selected_pairs, axis=1)
        selected_pairs = np.unique(selected_pairs, axis=0)
        binned_pairs.append(selected_pairs)

    return binned_pairs



if __name__ == "__main__":
    network_variables = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", True)
    rnn_interconnectivity = get_rnn_interconnectivity(network_variables)
    only_layer = rnn_interconnectivity[list(rnn_interconnectivity.keys())[0]]
    sc_pairs = get_strongly_connected_neuron_pairs(only_layer, 0.08, absolute_weights=True)
    binned_sc_pairs = group_neuron_pairs_by_connection_strength(only_layer, 0.02, absolute_weights=False)

    for sc_pairs in binned_sc_pairs:
        if sc_pairs.shape[0] == 0:
            pass
        else:
            d = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
            activity_pairs = get_activity_profiles_by_indices_pairs(d, sc_pairs)
            similarity, similarity_diff = compute_paired_similarity_metrics(activity_pairs)
            print(similarity)
            print(similarity_diff)
            print()

            comp_rnd_similarity = []
            comp_rnd_similarity_diff = []
            for i in range(10):
                shuffled_sc_pairs = copy.copy(sc_pairs)
                np.random.shuffle(shuffled_sc_pairs[:, 0])
                np.random.shuffle(shuffled_sc_pairs[:, 1])

                activity_pairs_shuffled = get_activity_profiles_by_indices_pairs(d, shuffled_sc_pairs)
                rnd_similarity, rnd_similarity_diff = compute_paired_similarity_metrics(activity_pairs_shuffled)
                comp_rnd_similarity.append(rnd_similarity)
                comp_rnd_similarity_diff.append(rnd_similarity_diff)

            print(np.mean(comp_rnd_similarity))
            print(np.mean(comp_rnd_similarity_diff))
            print()

    # plot_rnn_activity_pairs(activity_pairs)
    # compare_weights_to_and_from(only_layer)
    # plot_hist_all_weights(only_layer)
