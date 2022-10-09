import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces
from Analysis.Behavioural.Tools.BehavLabels.label_behavioural_context import label_capture_sequences


def display_activity_heat_map(rnn_activity, event_timestamps, name, min_t=0, max_t=-1, normalise_main_activity=True,
                              normalise_cell_activity=True):
    main_activity = np.swapaxes(rnn_activity[min_t:max_t, 0, 0, :], 0, 1)
    cell_state_activity = np.swapaxes(rnn_activity[min_t:max_t, 1, 0, :], 0, 1)

    if normalise_main_activity:
        main_activity = normalise_within_neuron_multiple_traces(main_activity)
    if normalise_cell_activity:
        cell_state_activity = normalise_within_neuron_multiple_traces(cell_state_activity)


    plt.figure(figsize=(20, 20))
    plt.pcolormesh(main_activity, cmap="bwr")
    for event in event_timestamps:
        plt.vlines(event, 0, main_activity.shape[0], color="yellow")
    plt.savefig(f"{name}-heat_map.jpg", dpi=100)
    plt.clf()
    plt.close()

    plt.figure(figsize=(20, 20))
    plt.pcolormesh(cell_state_activity, cmap="bwr")
    for event in event_timestamps:
        plt.vlines(event, 0, cell_state_activity.shape[0], color="yellow")
    plt.savefig(f"{name}-heat_map_cell.jpg", dpi=100)
    plt.clf()
    plt.close()


def display_activity_heat_map_capture_sequences_average(model_name, assay_config, assay_id, n,
                                                        normalise_main_activity=True, normalise_cell_activity=True,
                                                        sequence_steps=10):
    rnn_activity = np.zeros((sequence_steps, 2, 512))
    num_sequences = 0
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        capture_ts = label_capture_sequences(data, n=sequence_steps) * 1
        rnn_data = data["rnn_state_actor"]
        main_activity = np.swapaxes(rnn_data[:, 0:1, 0, :], 0, 1)
        cell_state_activity = np.swapaxes(rnn_data[:, 1:2, 0, :], 0, 1)
        if normalise_main_activity:
            main_activity = normalise_within_neuron_multiple_traces(main_activity)
        if normalise_cell_activity:
            cell_state_activity = normalise_within_neuron_multiple_traces(cell_state_activity)
        rnn_data = np.concatenate((main_activity, cell_state_activity), axis=0)
        rnn_data = np.swapaxes(rnn_data, 0, 1)

        capture_ts_separated = []
        current_ts_sequence = []
        for i, c in enumerate(capture_ts):
            if c == 1:
                current_ts_sequence.append(i)
            else:
                if len(current_ts_sequence) > 0:
                    capture_ts_separated.append(current_ts_sequence)
                    current_ts_sequence = []
        if len(current_ts_sequence) > 0:
            capture_ts_separated.append(current_ts_sequence)
            current_ts_sequence = []

        for sequence in capture_ts_separated:
            if len(sequence) != sequence_steps:
                continue
            rnn_activity_t = rnn_data[sequence]
            rnn_activity += rnn_activity_t
            num_sequences += 1
    rnn_activity /= num_sequences
    main_activity = np.swapaxes(rnn_activity[:, 0, :], 0, 1)
    cell_state_activity = np.swapaxes(rnn_activity[:, 1, :], 0, 1)


    plt.figure(figsize=(20, 20))
    plt.pcolormesh(main_activity, cmap="bwr")
    plt.savefig(f"{model_name}-prey_capture-heat_map.jpg", dpi=100)
    plt.clf()
    plt.close()

    plt.figure(figsize=(20, 20))
    plt.pcolormesh(cell_state_activity, cmap="bwr")

    plt.savefig(f"{model_name}-prey_capture-heat_map_cell.jpg", dpi=100)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    display_activity_heat_map_capture_sequences_average("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 10,
                                                        normalise_main_activity=False, normalise_cell_activity=False)
    # Interrupted trials
    # data = load_data("dqn_scaffold_14-1", "Interruptions-H", "Naturalistic-3")
    # data2 = load_data("dqn_scaffold_14-1", "Interruptions-H", "Naturalistic-5")
    # display_activity_heat_map(data2["rnn_state_actor"], [200], "Interruptions", max_t=1000)
    #
    # # Normal trials
    # data = load_data("ppo_scaffold_21-1", "Behavioural-Data-Free", "Naturalistic-5")
    # consumption_events = [i for i, c in enumerate(data["consumed"]) if c == 1]
    # display_activity_heat_map(data["rnn_state_actor"], consumption_events, "Normal Behaviour")

