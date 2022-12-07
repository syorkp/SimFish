import numpy as np

import matplotlib.pyplot as plt

from Analysis.load_data import load_data


def display_rnn_traces_many_trials(model_name, assay_config, assay_id, n):
    trials = []
    trial_lengths = []
    for i in range(1, n+1):
        if i == 11: continue
        else:
            d = load_data(model_name, assay_config, f"{assay_id}-{i}")
            trials.append(d["rnn_shared"][:, 0])
            trial_lengths.append(d["rnn_shared"].shape[0])

    min_trial_length = min(trial_lengths)

    neuron_traces = []
    n_neurons = 512
    for neur in range(n_neurons):
        for_neuron = []
        for trial in trials:
            for_neuron.append(trial[:, neur])
        neuron_traces.append(for_neuron)

    # for neur in range(n_neurons):
    #     for i in range(n):
    #         plt.plot(neuron_traces[neur][i], alpha=0.1)
    for i in range(n-1):
        for neur in range(n_neurons):
            plt.plot(neuron_traces[neur][i], alpha=0.1)
    # plt.xlim(0, min_trial_length)
    plt.xlabel("Step")
    plt.ylabel("Raw Unit Activity")

    plt.savefig(f"Plots/{i}-{model_name}-All_RNN-Units-{assay_config}.jpg")
    plt.close()
    plt.clf()


if __name__ == "__main__":
    display_rnn_traces_many_trials("dqn_new-2", "Behavioural-Data-Free", "Naturalistic", 20)
    # display_rnn_traces_many_trials("dqn_new-1", "Behavioural-Data-RNN-Zero", "Naturalistic", 20)
    # display_rnn_traces_many_trials("dqn_new-2", "Behavioural-Data-RNN-Zero", "Naturalistic", 20)



