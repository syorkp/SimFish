import matplotlib.pyplot as plt
import seaborn as sns

from Analysis.Visualisation.display_many_neurons import plot_traces
from Analysis.load_data import load_data
from Analysis.load_stimuli_data import load_stimulus_data


def plot_actions_and_activity(data, trace):
    action_choice = data["behavioural choice"]
    separated_actions = []
    fig, axs = plt.subplots(2,1, sharex=True)
    for action in range(max(action_choice)):
        action_timestamps = [i for i, a in enumerate(action_choice) if a == action]
        separated_actions.append(action_timestamps)
    colors = sns.color_palette("hls", max(action_choice))
    axs[0].plot(trace)
    axs[1].eventplot(separated_actions, color=colors)
    axs[0].set_ylabel("Unit activity")
    axs[1].set_ylabel("Action")
    axs[1].set_xlabel("Step")
    plt.show()


data1a = load_data("even_prey_ref-4", "Prey-Full-Response-Vector", "Prey-Left-15")
stimulus_data1 = load_stimulus_data("even_prey_ref-4", "Prey-Full-Response-Vector", "Prey-Left-15")
unit_activity1a = [[data1a["rnn state"][i - 1][0][j] for i in data1a["step"]] for j in range(512)]
activity = unit_activity1a[273]

plot_actions_and_activity(data1a, activity)

# plot_traces(unit_activity1a, stimulus_data1)
