import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


from Analysis.load_data import load_data
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces
from Analysis.Behavioural.Tools.BehavLabels.label_behavioural_context import label_capture_sequences


def display_activity_heat_map(rnn_activity, event_timestamps, name, min_t=0, max_t=-1, normalise_main_activity=True,
                              normalise_cell_activity=True, remove_undeviating=False, detrend=True):
    main_activity = np.swapaxes(rnn_activity[min_t:max_t, 0, 0, :], 0, 1)
    cell_state_activity = np.swapaxes(rnn_activity[min_t:max_t, 1, 0, :], 0, 1)

    if detrend:
        main_activity = signal.detrend(main_activity)
        cell_state_activity = signal.detrend(cell_state_activity)

    if normalise_main_activity:
        main_activity = normalise_within_neuron_multiple_traces(main_activity)
    if normalise_cell_activity:
        cell_state_activity = normalise_within_neuron_multiple_traces(cell_state_activity)

    if remove_undeviating:
        thresh_1 = 0.20
        # thresh_1 = 0.05
        # thresh_1 = 0.07
        std_main = np.std(main_activity[:, 200:], axis=1)
        varying_main = (std_main > thresh_1)
        main_activity = main_activity[varying_main]

        thresh_2 = 0.20
        # thresh_2 = 0.15
        # thresh_2 = 0.17
        std_cell = np.std(cell_state_activity[:, 200:], axis=1)
        varying_cell = (std_cell > thresh_2)
        cell_state_activity = cell_state_activity[varying_cell]

    plt.figure(figsize=(20, 5))
    plt.pcolormesh(main_activity, cmap="bwr")
    for event in event_timestamps:
        plt.vlines(event, 0, main_activity.shape[0], color="black")
    plt.ylabel("Neurons", fontsize=25)
    plt.xlabel("Step", fontsize=25)
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
                                                        sequence_steps=10, post_steps=0, first_captures=False,
                                                        remove_undeviating=True):
    all_rnn_activity = []
    all_capture_timepoints = []
    num_sequences = 0
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        capture_ts = label_capture_sequences(data, n=sequence_steps) * 1
        rnn_data = data["rnn_state_actor"]
        all_rnn_activity.append(rnn_data)
        all_capture_timepoints.append(capture_ts)

    all_capture_timepoints = np.concatenate(all_capture_timepoints)
    all_rnn_activity = np.concatenate(all_rnn_activity)

    # RNN normalisation
    main_activity = np.swapaxes(all_rnn_activity[:, 0, 0, :], 0, 1)
    cell_state_activity = np.swapaxes(all_rnn_activity[:, 1, 0, :], 0, 1)

    main_activity = signal.detrend(main_activity)
    cell_state_activity = signal.detrend(cell_state_activity)

    if normalise_main_activity:
        main_activity = normalise_within_neuron_multiple_traces(main_activity, zero_score_start=False)
    if normalise_cell_activity:
        cell_state_activity = normalise_within_neuron_multiple_traces(cell_state_activity, zero_score_start=False)
    rnn_data = np.concatenate((np.expand_dims(main_activity, 1),
                               np.expand_dims(cell_state_activity, 1)),
                              axis=1)
    rnn_data = np.swapaxes(rnn_data, 0, 2)

    # Selecting relevant capture points
    capture_ts_separated = []
    current_ts_sequence = []
    for i, c in enumerate(all_capture_timepoints):
        if c == 1:
            current_ts_sequence.append(i)
        else:
            if len(current_ts_sequence) > 0:
                capture_ts_separated.append(current_ts_sequence)
                current_ts_sequence = []
    if len(current_ts_sequence) > 0:
        capture_ts_separated.append(current_ts_sequence)
        current_ts_sequence = []

    # Adding more steps to each sequence
    for sequence in capture_ts_separated:
        sequence += [i for i in range(sequence[-1]+1, sequence[-1]+post_steps+1)]

    # Compiling activity during sequences
    rnn_activity = np.zeros((sequence_steps+post_steps, 2, 512))

    for sequence in capture_ts_separated:
        if len(sequence) != sequence_steps + post_steps:
            continue
        try:
            rnn_activity_t = rnn_data[sequence]
            rnn_activity += rnn_activity_t
            num_sequences += 1
        except IndexError:
            pass

    # Changing to mean
    rnn_activity /= num_sequences
    main_activity = np.swapaxes(rnn_activity[:, 0, :], 0, 1)
    cell_state_activity = np.swapaxes(rnn_activity[:, 1, :], 0, 1)

    if remove_undeviating:
        thresh_1 = 0.07
        # thresh_1 = 0.05
        # thresh_1 = 0.07
        std_main = np.std(main_activity, axis=1)
        varying_main = (std_main > thresh_1)
        main_activity = main_activity[varying_main]

        thresh_2 = 0.20
        # thresh_2 = 0.15
        # thresh_2 = 0.17
        std_cell = np.std(cell_state_activity, axis=1)
        varying_cell = (std_cell > thresh_2)
        cell_state_activity = cell_state_activity[varying_cell]

    time_ticks = [i for i in range(-sequence_steps, 1)] + [i for i in range(1, post_steps + 1)]

    # Plotting
    plt.figure(figsize=(20, 20))
    plt.pcolormesh(main_activity, cmap="bwr")
    plt.vlines(sequence_steps, 0, main_activity.shape[0], color="black")
    plt.xticks(range(0, len(time_ticks), 5), [time_ticks[i] for i in range(0, len(time_ticks), 5)], fontsize=15)
    plt.ylabel("Neurons", fontsize=25)
    plt.xlabel("Steps from Consumption", fontsize=25)
    plt.savefig(f"{model_name}-prey_capture-heat_map_main.jpg", dpi=100)
    plt.clf()
    plt.close()

    plt.figure(figsize=(20, 20))
    plt.pcolormesh(cell_state_activity, cmap="bwr")
    plt.vlines(sequence_steps, 0, cell_state_activity.shape[0], color="black")
    plt.xticks(range(0, len(time_ticks), 5), [time_ticks[i] for i in range(0, len(time_ticks), 5)], fontsize=15)
    plt.ylabel("Neurons", fontsize=25)
    plt.xlabel("Steps from Consumption", fontsize=25)
    plt.savefig(f"{model_name}-prey_capture-heat_map_cell.jpg", dpi=100)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    # display_activity_heat_map_capture_sequences_average("dqn_scaffold_26-2", "Behavioural-Data-NaturalisticA", "Naturalistic", 20,
    #                                                     normalise_main_activity=True, normalise_cell_activity=True,
    #                                                     sequence_steps=50, post_steps=50)
    # Interrupted trials
    # data = load_data("dqn_scaffold_14-1", "Interruptions-HA", "Naturalistic-3")
    data2 = load_data("dqn_new-2", "Behavioural-Data-Free", "Naturalistic-1")
    rnn_data = np.expand_dims(data2["rnn_shared"], 1)
    rnn_data = np.concatenate((rnn_data, rnn_data), axis=1)
    display_activity_heat_map(rnn_data, [200], "dqn_new-1", max_t=2000, remove_undeviating=False, detrend=False)
    #
    # # Normal trials
    # data = load_data("ppo_scaffold_21-2", "Behavioural-Data-Videos-A1", "Naturalistic-5")
    # consumption_events = [i for i, c in enumerate(data["consumed"]) if c == 1]
    # display_activity_heat_map(data["rnn_state_actor"], consumption_events, "ppo_scaffold_21-2-Normal Behaviour")
    #
    # data = load_data("dqn_scaffold_26-2", "Behavioural-Data-Videos-A1", "Naturalistic-1")
    # consumption_events = [i for i, c in enumerate(data["consumed"]) if c == 1]
    # display_activity_heat_map(data["rnn_state_actor"], consumption_events, "dqn_scaffold_26-2-Normal Behaviour")


