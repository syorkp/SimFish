import math
from matplotlib import pyplot as plt
from Analysis.load_data import load_data
from Analysis.load_stimuli_data import load_stimulus_data


def create_plot(neuron_data, plot_number, n_subplots, n_prev_subplots, stimulus_data=None):
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots / 2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"Neuron group {plot_number}", fontsize=16)

    for i in range(int(n_subplots / 2)):
        axs[i, 0].plot(neuron_data[i])
        if stimulus_data:
            for period in stimulus_data:
                for key in period.keys():
                    axs[i, 0].hlines(y=min(neuron_data[i]) - 1, xmin=period[key]["Onset"],
                                     xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                                     color="r")
        axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} activity")
        axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots / 2)):
        axs[i, 1].plot(neuron_data[i + int(n_subplots / 2)])
        if stimulus_data:
            for period in stimulus_data:
                for key in period.keys():
                    axs[i, 1].hlines(y=min(neuron_data[i + int(n_subplots / 2)]) - 1, xmin=period[key]["Onset"],
                                     xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                                     color="r")
        axs[i, 1].set_ylabel(f"Unit {i + int(n_subplots / 2) + n_prev_subplots} activity")
        axs[i, 1].tick_params(labelsize=15)

    # Add graph annotations
    axs[int(n_subplots / 2 - 1), 0].set_xlabel("Step", fontsize=25)

    axs[int(n_subplots / 2 - 1), 1].set_xlabel("Step", fontsize=25)

    fig.set_size_inches(18.5, 20)
    plt.show()


def create_plot_multiple_traces(neuron_data_traces, plot_number, n_subplots, n_prev_subplots, stimulus_data=None, trace_names=None):
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots / 2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"Neuron group {plot_number}", fontsize=16)

    for i in range(int(n_subplots / 2)):
        for j, trace in enumerate(neuron_data_traces):
            if trace_names:
                axs[i, 0].plot(trace[i], label=trace_names[j])
            else:
                axs[i, 0].plot(trace[i])
        if stimulus_data[0]:
            for period in stimulus_data[0]:
                for key in period.keys():
                    axs[i, 0].hlines(y=min(neuron_data_traces[0][i]) - 1,
                                     xmin=period[key]["Onset"],
                                     xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                                     color="r")
        axs[i, 0].set_ylabel(f"Unit {i + n_prev_subplots} activity")
        axs[i, 0].tick_params(labelsize=15)

    for i in range(int(n_subplots / 2)):
        for j, trace in enumerate(neuron_data_traces):
            if trace_names:
                axs[i, 1].plot(trace[i + int(n_subplots / 2)], label=trace_names[j])
            else:
                axs[i, 1].plot(trace[i + int(n_subplots / 2)])
        if stimulus_data[0]:
            for period in stimulus_data[0]:
                for key in period.keys():
                    axs[i, 1].hlines(y=min(neuron_data_traces[0][i + int(n_subplots / 2)]) - 1,
                                     xmin=period[key]["Onset"],
                                     xmax=period[key]["Onset"] + (period[key]["Onset"] - period[key]["Pre-onset"]),
                                     color="r")
        axs[i, 1].set_ylabel(f"Unit {i + int(n_subplots / 2) + n_prev_subplots} activity")
        axs[i, 1].tick_params(labelsize=15)
    if trace_names:
        axs[0, 1].legend(bbox_to_anchor=(1, 3), loc='upper right')

    # Add graph annotations
    axs[int(n_subplots / 2 - 1), 0].set_xlabel("Step", fontsize=25)

    axs[int(n_subplots / 2 - 1), 1].set_xlabel("Step", fontsize=25)

    fig.set_size_inches(18.5, 20)
    plt.savefig(f"Neuron group {plot_number}")
    plt.show()


def plot_traces(neuron_data, stimulus_data=None):
    n_subplots = len(neuron_data)
    n_per_plot = 30
    n_plots = math.ceil(n_subplots / n_per_plot)

    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data = neuron_data[(i * n_per_plot):]
        else:
            neuron_subset_data = neuron_data[(i * n_per_plot): (i * n_per_plot) + n_per_plot]
        create_plot(neuron_subset_data, i + 1, len(neuron_subset_data), i * n_per_plot, stimulus_data)


def plot_multiple_traces(neuron_data_list, stimulus_data_list=None, trace_names=None):
    n_subplots = len(neuron_data_list[0])
    n_per_plot = 30
    n_plots = math.ceil(n_subplots / n_per_plot)

    for i in range(n_plots):
        if i == n_plots - 1:
            neuron_subset_data = [neuron_data[(i * n_per_plot):] for neuron_data in neuron_data_list]
        else:
            neuron_subset_data = [neuron_data[(i * n_per_plot): (i * n_per_plot) + n_per_plot] for neuron_data in
                                  neuron_data_list]
        create_plot_multiple_traces(neuron_subset_data, i + 1, len(neuron_subset_data[0]), i * n_per_plot,
                                    stimulus_data_list, trace_names=trace_names)



def get_free_swimming_timestamps(data):
    """Requires the following data: position, prey_positions, predator. Assumes square arena 1500."""
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    wall_timestamps = [i for i, p in enumerate(data["position"]) if 200 < p[0] < 1300 and 200<p[1]<1300]
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


def plot_behavioural_events(data, duration=None):
    # fig, axs = plt.figure()
    plt.figure()
    if duration:
        plt.xlim(0, duration)
    else:
        plt.xlim(0, max(data["step"]))
    plt.ylim(-3, 3)
    predator_steps = [step for step, predator in zip(data["step"], data["predator"]) if predator == 1]
    free_steps = get_free_swimming_timestamps(data)
    separated_predator_steps = []
    current_sequence = []
    for i, step in enumerate(predator_steps):
        if i == 0:
            current_sequence.append(step)
            continue
        if step -1 == predator_steps[i-1]:
            current_sequence.append(step)
        else:
            separated_predator_steps.append(current_sequence)
            current_sequence = []
            current_sequence.append(step)

    separated_predator_steps.append(current_sequence)
    current_sequence = []

    separated_free_steps = []
    current_sequence = []
    for i, step in enumerate(free_steps):
        if i == 0:
            current_sequence.append(step)
            continue
        if step -1 == free_steps[i-1]:
            current_sequence.append(step)
        else:
            separated_free_steps.append(current_sequence)
            current_sequence = []
            current_sequence.append(step)
    separated_free_steps.append(current_sequence)
    current_sequence = []

    # plt.plot(data["rnn state"][2][0])
    if len(data["consumed"]) > 0:
        for step, consumed in zip(data["step"], data["consumed"]):
            if consumed:
                plt.axvline(step, -1, 1, color="r")

    for steps in separated_predator_steps:
        plt.hlines(y=0, xmin=steps[0], xmax=steps[-1], color="r")

    for steps in separated_free_steps:
        plt.hlines(y=0, xmin=steps[0], xmax=steps[-1], color="g")

    plt.show()

import seaborn as sns

def plot_certain_neurons(data, duration, neuron_list):
    sns.set()
    fig, axs = plt.subplots(len(neuron_list)+1, 1, sharex=True)
    for i, n in enumerate(neuron_list):
        print(i)
        axs[i].plot(data["rnn state"][:duration, 0, n])
        axs[i].set_xlim(0, duration)
        axs[i].set_yticks([])

    predator_steps = [step for step, predator in zip(data["step"], data["predator"]) if predator == 1]
    free_steps = get_free_swimming_timestamps(data)
    separated_predator_steps = []
    current_sequence = []
    for i, step in enumerate(predator_steps):
        if i == 0:
            current_sequence.append(step)
            continue
        if step - 1 == predator_steps[i - 1]:
            current_sequence.append(step)
        else:
            separated_predator_steps.append(current_sequence)
            current_sequence = []
            current_sequence.append(step)

    separated_predator_steps.append(current_sequence)
    current_sequence = []

    separated_free_steps = []
    current_sequence = []
    for i, step in enumerate(free_steps):
        if i == 0:
            current_sequence.append(step)
            continue
        if step - 1 == free_steps[i - 1]:
            current_sequence.append(step)
        else:
            separated_free_steps.append(current_sequence)
            current_sequence = []
            current_sequence.append(step)
    separated_free_steps.append(current_sequence)
    current_sequence = []

    # plt.plot(data["rnn state"][2][0])
    if len(data["consumed"]) > 0:
        for step, consumed in zip(data["step"], data["consumed"]):
            if consumed:
                axs[-1].axvline(step, -1, 1, color="r")
    axs[-1].set_xlim(0, duration)
    axs[-1].set_yticks([])

    for steps in separated_predator_steps:
        axs[-1].hlines(y=0, xmin=steps[0], xmax=steps[-1], color="r")

    for steps in separated_free_steps:
        axs[-1].hlines(y=0, xmin=steps[0], xmax=steps[-1], color="g")
    fig.set_size_inches((8, 8))
    axs[-1].set_xlabel("Time (steps)")
    plt.show()


def plot_artificial_traces(prey_pred_data, prey_size_data, directional_data, prey_pred_neurons, prey_size_neurons, directional_neurons,
                           labels1, labels2, labels3, stimulus_data1, stimulus_data2, stimulus_data3):
    sns.set()
    fig, axs = plt.subplots(len(prey_pred_neurons+prey_size_neurons+directional_neurons), 1, sharex=True)
    current_i = 0
    for neuron in prey_pred_neurons:
        axs[current_i].plot(prey_pred_data[0][neuron])
        axs[current_i].plot(prey_pred_data[1][neuron])
        current_i += 1
    for neuron in prey_size_neurons:
        axs[current_i].plot(prey_size_data[0][neuron])
        axs[current_i].plot(prey_size_data[1][neuron])
        axs[current_i].plot(prey_size_data[2][neuron])
        current_i += 1
    for neuron in directional_neurons:
        axs[current_i].plot(directional_data[0][neuron])
        axs[current_i].plot(directional_data[1][neuron])
        current_i += 1
    plt.show()



# data1 = load_data("new_differential_prey_ref-4", "Behavioural-Data-Free-1", "Naturalistic-8")
# plot_certain_neurons(data1, 200, [20, 21, 139, 156, 161, 135, 133, 196])
# plot_behavioural_events(data1, 200)
# # Get exploraiton timeseries and add to that
# unit_activity1 = [[data1["rnn state"][i - 1][0][j] for i in range(200)] for j in range(512)]
# plot_traces(unit_activity1)




data1 = load_data("even_prey_ref-5", "For-Traces", "Prey-Static-10")
data2 = load_data("even_prey_ref-5", "For-Traces", "Predator-Static-40")
stimulus_data1 = load_stimulus_data("even_prey_ref-5", "For-Traces", "Prey-Static-10")
stimulus_data2 = load_stimulus_data("even_prey_ref-5", "For-Traces", "Predator-Static-40")

unit_activity1 = [[data1["rnn state"][i - 1][0][j] for i in data1["step"]] for j in range(512)]
unit_activity2 = [[data2["rnn state"][i - 1][0][j] for i in data2["step"]] for j in range(512)]

#
# plot_multiple_traces([unit_activity1, unit_activity2],
#                      [stimulus_data1, stimulus_data2],
#                      ["Prey", "Predator"])


data1 = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-5")
data2 = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-10")
data3 = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-15")
stimulus_data3 = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-5")
stimulus_data4 = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-10")
stimulus_data5 = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-15")

unit_activity3 = [[data1["rnn state"][i - 1][0][j] for i in data1["step"]] for j in range(512)]
unit_activity4 = [[data2["rnn state"][i - 1][0][j] for i in data2["step"]] for j in range(512)]
unit_activity5 = [[data3["rnn state"][i - 1][0][j] for i in data3["step"]] for j in range(512)]
# plot_multiple_traces([unit_activity3, unit_activity4, unit_activity5],
#                      [stimulus_data3, stimulus_data4, stimulus_data5],
#                      ["5", "10", "15"])

data1 = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Left-10")
data2 = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Right-10")
stimulus_data6 = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-5")
stimulus_data7 = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-10")
unit_activity6 = [[data1["rnn state"][i - 1][0][j] for i in data1["step"]] for j in range(512)]
unit_activity7 = [[data2["rnn state"][i - 1][0][j] for i in data2["step"]] for j in range(512)]
# plot_multiple_traces([unit_activity6, unit_activity7],
#                      [stimulus_data1, stimulus_data2],
#                      ["Left", "Right"])


plot_artificial_traces([unit_activity1, unit_activity2], [unit_activity3, unit_activity4, unit_activity5], [unit_activity6, unit_activity7],
                       [158, 315, 319], [21, 152, 315, 302], [335, 302, 315],
                       ["Prey-10", "Predator-40"], ["Prey-5", "Prey-10", "Prey-15"], ["Prey-10-Left", "Prey-10-Right"],
                       [stimulus_data1, stimulus_data2], [stimulus_data3, stimulus_data4, stimulus_data5], [stimulus_data6, stimulus_data7])

# data1a = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-5")
# data2b = load_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Static-40")
# data1a = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Towards")
# data1b = load_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Away")
# data2a = load_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Towards")
# data2b = load_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Away")
#
# # stimulus_data1a = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Static-5")
# # stimulus_data2a = load_stimulus_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Static-40")
# stimulus_data1a = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Towards")
# stimulus_data1b = load_stimulus_data("even_prey_ref-5", "Prey-Full-Response-Vector", "Prey-Away")
# stimulus_data2a = load_stimulus_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Towards")
# stimulus_data2b = load_stimulus_data("even_prey_ref-5", "Predator-Full-Response-Vector", "Predator-Away")


# unit_activity1a = [[data1a["rnn state"][i - 1][0][j] for i in data1a["step"]] for j in range(512)]
# unit_activity1b = [[data1b["rnn state"][i - 1][0][j] for i in data2b["step"]] for j in range(512)]
# unit_activity2a = [[data2a["rnn state"][i - 1][0][j] for i in data1a["step"]] for j in range(512)]
# unit_activity2b = [[data2b["rnn state"][i - 1][0][j] for i in data2b["step"]] for j in range(512)]

# Many presentations for RF mapping (Prey)
# plot_multiple_traces([unit_activity1a, unit_activity1b, unit_activity2a, unit_activity2b],
#                      [stimulus_data1a, stimulus_data1b, stimulus_data2a, stimulus_data2b],
#                      ["Prey-Towards", "Prey-Away", "Predator-Towards", "Predator-Away"])


# No vs red background
# plot_multiple_traces([unit_activity1a, unit_activity1],
#                      [stimulus_data1a, stimulus_data1],
#                      ["No Background", "Red Background"])

# Left vs right predators
# plot_multiple_traces([unit_activity2, unit_activity3],
#                      [stimulus_data2, stimulus_data3],
#                      ["Predator-Moving-Left", "Predator-Moving-Right"])

# Left vs right prey
# plot_multiple_traces([unit_activity5, unit_activity6],
#                      [stimulus_data5, stimulus_data6],
#                      ["Prey-Moving-Left", "Prey-Moving-Right"])

# # Static prey vs predators
# plot_multiple_traces([unit_activity1, unit_activity4],
#                      [stimulus_data1, stimulus_data4],
#                      ["Predator-Static", "Prey-Static"])
#
#
# Right prey vs static
# plot_multiple_traces([unit_activity4, unit_activity6],
#                      [stimulus_data4, stimulus_data6],
#                      ["Prey-Static", "Prey-Moving-Right"])

# # Prey of different sizes
# plot_multiple_traces([unit_activity7, unit_activity8, unit_activity9],
#                      [stimulus_data7, stimulus_data8, stimulus_data9],
#                      ["Prey-Size-5", "Prey-Size-10", "Prey-Size-15"])

# All
# plot_multiple_traces([unit_activity1, unit_activity2, unit_activity3, unit_activity4, unit_activity5, unit_activity6],
#                      [stimulus_data1, stimulus_data2, stimulus_data3, stimulus_data4, stimulus_data5, stimulus_data6],
#                      ["Predator-Static", "Predator-Moving-Left", "Predator-Moving-Right", "Prey-Static", "Prey-Moving-Left", "Prey-Moving-Right"])
