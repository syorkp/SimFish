import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron_multiple_traces


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


if __name__ == "__main__":
    # Interrupted trials
    data = load_data("dqn_scaffold_14-1", "Interruptions-H", "Naturalistic-3")
    data2 = load_data("dqn_scaffold_14-1", "Interruptions-H", "Naturalistic-5")
    display_activity_heat_map(data2["rnn_state_actor"], [200], "Interruptions", max_t=1000)

    # Normal trials
    data = load_data("ppo_scaffold_21-1", "Behavioural-Data-Free", "Naturalistic-5")
    consumption_events = [i for i, c in enumerate(data["consumed"]) if c == 1]
    display_activity_heat_map(data["rnn_state_actor"], consumption_events, "Normal Behaviour")

