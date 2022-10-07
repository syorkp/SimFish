import copy

import numpy as np
import matplotlib.pyplot as plt

from Analysis.Connectivity.load_network_variables import load_network_variables_dqn
from Analysis.load_data import load_data
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces

import tensorflow.compat.v1 as tf


def get_rnn_interconnectivity(all_network_variables, gate_num=2, rnn_num=512):
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
    if gate_num is None:
        selected_weights = {key: rnn_related[key] for key in rnn_related.keys()}
    else:
        gate_index_start = rnn_num * gate_num
        selected_weights = {key: rnn_related[key][:, gate_index_start:gate_index_start+rnn_num] for key in rnn_related.keys()}
    return selected_weights


def plot_hist_all_weights(all_weights_1, all_weights_2=None, name="unspecified", bins=100):
    all_weights_1 = all_weights_1.flatten()

    plt.hist(all_weights_1, bins=bins, alpha=0.5)

    if all_weights_2 is not None:
        all_weights_2 = all_weights_2.flatten()
        plt.hist(all_weights_2, bins=bins, alpha=0.5)
    plt.savefig(f"{name}-hist_all_weights.png")
    plt.clf()


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
    plt.clf()


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
    plt.clf()


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

    return binned_pairs, bins


def plot_mse_across(bins, mse, mse_diff, mse_random, mse_diff_random):
    # bins = np.array([-0.11949778, -0.09839867, -0.07729957, -0.05620046,
    #                  -0.03510136, -0.01400225,  0.00709685,  0.02819595,  0.04929506,  0.07039416,  0.09149327,  0.11259237,
    #                  0.13369148,])

    # mse = [0.14077806, 0.28180635, 0.44530493, 0.61818, 0.63800067, 0.65239006, 0.6836712, 0.68562895, 0.6042702, 0.55133086],
    # mse_random = [0.14077806, 0.2597273, 0.46050373, 0.61027795, 0.638971,  0.64753425, 0.6749977, 0.69010156, 0.58166707, 0.5513309,]
    # mse_diff = [0.005095943, 0.008683029, 0.0037848349, 0.0020505749, 0.001801332, 0.0016352658, 0.0013567825, 0.0012184646, 0.0013878962, 0.0040659336]
    # mse_diff_random = [0.005095943, 0.0069771274, 0.0037635076, 0.0020360008, 0.0017907234, 0.0016454477,0.0013632916,0.0012090596,0.001801214, 0.0040659336]

    bin_edges = np.array(bins)[:, 0]

    plt.plot(bin_edges, mse)
    plt.plot(bin_edges, mse_random)
    plt.savefig("MSE bins.png")
    plt.clf()

    plt.plot(bin_edges, mse_diff)
    plt.plot(bin_edges, mse_diff_random)
    plt.savefig("MSE diff bins.png")
    plt.clf()

    x = True


if __name__ == "__main__":
    # with open('ablatedValues.npy', 'rb') as f:
    #     ablated = np.load(f)
    # PPO
    with open("ppo21_2_rnn.npy", "rb") as f:
        network_variables_ppo = np.load(f)
    plot_hist_all_weights(network_variables_ppo, name="ppo_21-2-all", bins=2000)
    plot_hist_all_weights(network_variables_ppo[:512, 512:1024], name="ppo_21-2-selected", bins=2000)


    # DQN untrained model
    network_variables_2 = load_network_variables_dqn("dqn_scaffold_26-5", "dqn_26_2", True)
    rnn_interconnectivity_1 = get_rnn_interconnectivity(network_variables_2, gate_num=None)
    rnn_interconnectivity_1 = rnn_interconnectivity_1["main_rnn/lstm_cell/kernel:0"]
    plot_hist_all_weights(rnn_interconnectivity_1, name="untrained-full", bins=2000)

    tf.reset_default_graph()

    # DQN
    network_variables_1 = load_network_variables_dqn("dqn_scaffold_14-1", "dqn_14_1", False)
    rnn_interconnectivity_1 = get_rnn_interconnectivity(network_variables_1, gate_num=None)
    only_layer_1 = rnn_interconnectivity_1[list(rnn_interconnectivity_1.keys())[0]]
    tf.reset_default_graph()

    network_variables_2 = load_network_variables_dqn("dqn_scaffold_26-2", "dqn_26_2", True)
    rnn_interconnectivity_2 = get_rnn_interconnectivity(network_variables_2, gate_num=1)
    only_layer_2 = rnn_interconnectivity_2[list(rnn_interconnectivity_2.keys())[0]]

    rnn_interconnectivity_1 = rnn_interconnectivity_1["main_rnn/lstm_cell/kernel:0"]
    selected_weights = rnn_interconnectivity_1[:512, 512:1024]
    to_include = 4 * np.std(selected_weights)
    print(np.sum((np.absolute(selected_weights) > to_include) * 1))
    selected_weights *= (np.absolute(selected_weights) > to_include)
    rnn_interconnectivity_1[:, 512:1024] = 0#selected_weights
    with open('../../Configurations/Ablation-Matrices/post_ablation_weights_2_dqn_14_1.npy', 'wb') as f:
        np.save(f, rnn_interconnectivity_1)


    # rnn_interconnectivity_2 = rnn_interconnectivity_2["main_rnn/lstm_cell/kernel:0"][:512]


    # Plot all Hist weights
    # plot_hist_all_weights(rnn_interconnectivity_1, name="dqn_scaffold_14-1", bins=2000)
    # plot_hist_all_weights(rnn_interconnectivity_2, name="dqn_scaffold_26-2", bins=2000)

    # compare_weights_to_and_from(only_layer_1)
    #
    # # Strongly connected unit weights
    # sc_pairs = get_strongly_connected_neuron_pairs(only_layer_1, 0.08, absolute_weights=True)
    # binned_sc_pairs, bins = group_neuron_pairs_by_connection_strength(only_layer_1, 0.02, absolute_weights=False)
    #
    # bins = [[bins[i], bins[i+1]] for i, b in enumerate(binned_sc_pairs) if b.shape[0] > 0]
    #
    # mse = []
    # mse_diff = []
    # rnd_mse = []
    # rnd_mse_diff = []
    # for sc_pairs in binned_sc_pairs:
    #     if sc_pairs.shape[0] == 0:
    #         pass
    #     else:
    #         d = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
    #         activity_pairs = get_activity_profiles_by_indices_pairs(d, sc_pairs)
    #         similarity, similarity_diff = compute_paired_similarity_metrics(activity_pairs)
    #         print(similarity)
    #         print(similarity_diff)
    #         print()
    #         mse.append(similarity)
    #         mse_diff.append(similarity_diff)
    #
    #
    #         comp_rnd_similarity = []
    #         comp_rnd_similarity_diff = []
    #         for i in range(10):
    #             shuffled_sc_pairs = copy.copy(sc_pairs)
    #             np.random.shuffle(shuffled_sc_pairs[:, 0])
    #             np.random.shuffle(shuffled_sc_pairs[:, 1])
    #
    #             activity_pairs_shuffled = get_activity_profiles_by_indices_pairs(d, shuffled_sc_pairs)
    #             rnd_similarity, rnd_similarity_diff = compute_paired_similarity_metrics(activity_pairs_shuffled)
    #             comp_rnd_similarity.append(rnd_similarity)
    #             comp_rnd_similarity_diff.append(rnd_similarity_diff)
    #
    #         print(np.mean(comp_rnd_similarity))
    #         print(np.mean(comp_rnd_similarity_diff))
    #         print()
    #
    #         rnd_mse.append(np.mean(comp_rnd_similarity))
    #         rnd_mse_diff.append(np.mean(comp_rnd_similarity_diff))
    #
    # plot_mse_across(bins, mse, mse_diff, rnd_mse, rnd_mse_diff)
    #
    # # plot_rnn_activity_pairs(activity_pairs)

