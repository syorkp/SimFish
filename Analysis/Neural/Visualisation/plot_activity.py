import json
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from Analysis.load_data import load_data


def plot_activity(rnn_data, stimulus_data, start_plot=None):
    number_of_neurons = len(rnn_data)
    n_subplots = number_of_neurons
    n_per_plot = 4
    n_plots = math.ceil(n_subplots / n_per_plot)

    for i in range(n_plots):
        fig, axs = plt.subplots(4, 1, sharex=True)
        if start_plot:
            plt.xlim(start_plot, len(rnn_data[0]))
        neuron = i * n_per_plot
        axs[0].plot(rnn_data[neuron])
        axs[0].set_ylabel(f"Unit {neuron} activity", fontsize=25)
        for period in stimulus_data:
            for key in period.keys():
                axs[0].hlines(y=min(rnn_data[neuron]) - 1,
                             xmin=period[key]["Onset"],
                             xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                             color="r", linewidth=5
                              )
        for period in stimulus_data:
            for key in period.keys():
                axs[0].hlines(y=min(rnn_data[neuron]) - 1,
                              xmin=period[key]["Onset"] - (period[key]["Onset"] - period[key]["Pre-onset"]),
                              xmax=period[key]["Onset"] - (period[key]["Onset"] - period[key]["Pre-onset"]) + (period[key]["Onset"] - period[key]["Pre-onset"]),
                              color="g", linewidth=5
                              )

        neuron += 1
        axs[1].plot(rnn_data[neuron])
        axs[1].set_ylabel("Unit 2 activity", fontsize=25)
        for period in stimulus_data:
            for key in period.keys():
                axs[1].hlines(y=min(rnn_data[neuron]) - 1,
                             xmin=period[key]["Onset"],
                             xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                             color="r", linewidth=5
                              )
        for period in stimulus_data:
            for key in period.keys():
                axs[1].hlines(y=min(rnn_data[neuron]) - 1,
                              xmin=period[key]["Onset"] - (period[key]["Onset"] - period[key]["Pre-onset"]),
                              xmax=period[key]["Onset"] - (period[key]["Onset"] - period[key]["Pre-onset"]) + (period[key]["Onset"] - period[key]["Pre-onset"]),
                              color="g", linewidth=5
                              )


        neuron += 1
        axs[2].plot(rnn_data[neuron])
        axs[2].set_ylabel("Unit 3 activity", fontsize=25)
        for period in stimulus_data:
            for key in period.keys():
                axs[2].hlines(y=min(rnn_data[neuron]) - 1,
                             xmin=period[key]["Onset"],
                             xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                             color="r", linewidth=5
                              )
        for period in stimulus_data:
            for key in period.keys():
                axs[2].hlines(y=min(rnn_data[neuron]) - 1,
                              xmin=period[key]["Onset"] - (period[key]["Onset"] - period[key]["Pre-onset"]),
                              xmax=period[key]["Onset"] - (period[key]["Onset"] - period[key]["Pre-onset"]) + (period[key]["Onset"] - period[key]["Pre-onset"]),
                              color="g", linewidth=5
                              )


        neuron += 1
        axs[3].plot(rnn_data[neuron])
        axs[3].set_ylabel("Unit 4 activity", fontsize=25)
        axs[3].set_xlabel("Step", fontsize=25)
        for period in stimulus_data:
            for key in period.keys():
                axs[3].hlines(y=min(rnn_data[neuron]) - 1,
                             xmin=period[key]["Onset"],
                             xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                             color="r", linewidth=5
                              )
        for period in stimulus_data:
            for key in period.keys():
                axs[3].hlines(y=min(rnn_data[neuron]) - 1,
                              xmin=period[key]["Onset"] - (period[key]["Onset"] - period[key]["Pre-onset"]),
                              xmax=period[key]["Onset"] - (period[key]["Onset"] - period[key]["Pre-onset"]) + (period[key]["Onset"] - period[key]["Pre-onset"]),
                              color="g", linewidth=5
                              )


        axs[0].tick_params(labelsize=15)
        axs[1].tick_params(labelsize=15)
        axs[2].tick_params(labelsize=15)
        axs[3].tick_params(labelsize=15)

        # axs[0].set_ylim(0.5, 1.5)
        fig.set_size_inches(18.5, 20)
        fig.savefig(f'../../Figures/Panels/x-{i}.png', dpi=100)
        plt.show()


def plot_activity_visual_presentation(model_name, assay_config, assay_id, units, start_plot=None):
    data = load_data("dqn_scaffold_14-1", "Prey-Full-Response-Vector", "Prey-Static-15")
    rnn_data = [[state[0, 0, j] for i, state in enumerate(data["rnn_state_actor"])] for j in range(512)]
    chosen_units = [unit for i, unit in enumerate(unit_activity1a) if i in units]
    stimulus_data = load_stimulus_data("dqn_scaffold_14-1", "Prey-Full-Response-Vector", "Prey-Static-15")


    number_of_neurons = len(rnn_data)

    new_rnn_data = []
    # Normalisation
    for rnn_d in rnn_data:
        max_d = max(rnn_d)
        min_d = min(rnn_d)
        range_d = max_d - min_d
        new_rnn_data.append(np.array(rnn_d) * (1/range_d))

    fig, axs = plt.subplots(figsize=(10, 8))

    for i in range(number_of_neurons):
        axs.plot(new_rnn_data[i][start_plot:] + i)

    for i, period in enumerate(stimulus_data):
        if i < 4:
            continue
        for key in period.keys():
            axs.hlines(y=-0.5,
                       xmin=period[key]["Onset"]-start_plot,
                       xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]) - start_plot,
                       color="r", linewidth=5
                        )

        # axs[0].set_ylim(0.5, 1.5)
    plt.yticks([])
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    plt.ylabel("LSTM Unit Activity (Normalised)", fontsize=25)
    plt.xlabel("Step", fontsize=25)
    axs.tick_params(axis="x", labelsize=20)
    axs.tick_params(axis="y", labelsize=20)
    plt.savefig(f'../../Figures/Panels/selected_visual_neurons.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    # Various visual neurons
    plot_activity_visual_presentation("dqn_scaffold_14-1", "Prey-Full-Response-Vector", "Prey-Static-15", [188, 192, 194, 234, 222], 600)

