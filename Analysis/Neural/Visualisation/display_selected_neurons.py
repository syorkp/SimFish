import json
import matplotlib.pyplot as plt

from Analysis.Neural.Visualisation.plot_activity import plot_activity


from Analysis.Neural.Visualisation.display_many_neurons import plot_traces
from Analysis.load_data import load_data
from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.Behavioural.Tools.extract_exploration_sequences import extract_exploration_action_sequences_with_positions


def display_neurons_with_selectivities(model_name, naturalistic, selectivity=False, action_association=False):

    with open(f"./../../Data/Categorisation-Data/{model_name}_categories.json", "r") as file:
        categories = json.load(file)

    if naturalistic:
        data = load_data(model_name, "Behavioural-Data-Free", "Naturalistic-4")
    else:
        selectivity_parts = selectivity.split("-")
        if len(selectivity_parts) > 4:
            selectivity_prefix = selectivity_parts[0] + "-" + selectivity_parts[1] + "-" + selectivity_parts[2]
        else:
            selectivity_prefix = selectivity_parts[0] + "-" + selectivity_parts[1]

        data = load_data(model_name, "Prey-Full-Response-Vector", selectivity_prefix)
        stimulus_data = load_stimulus_data(model_name, "Prey-Full-Response-Vector", selectivity_prefix)
    rnn_data = data["rnn_state_actor"][:, 0, 0, :]

    if action_association:
        action_associations = [i for i, key in enumerate(categories.keys()) if str(action_association) in categories[key]["action_associations"]]
        associated_neurons = action_associations
    else:
        associated_neurons = [i for i, key in enumerate(categories.keys())]

    if selectivity:
        visual_associations = [i for i, key in enumerate(categories.keys()) if selectivity in categories[key]["selectivities"]]
        associated_neurons = [i for i in visual_associations if i in associated_neurons]

    chosen_neurons = [rnn_data[:, i] for i in associated_neurons]

    if naturalistic:
        plot_traces(chosen_neurons)
    else:
        plot_activity(chosen_neurons, stimulus_data, start_plot=600)

import numpy as np


actions = load_data("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic-4")["action"]
event_data = np.zeros((10, len(actions)))
color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "k", "m", "m", "black"]

event_data = [[] for i in range(10)]

for i, a in enumerate(actions):
    event_data[a].append(i)


data = load_data(f"dqn_scaffold_14-1", "Behavioural-Data-Free", f"Naturalistic-4")
unit_activity1a = [[state[0, 0, j] for i, state in enumerate(data["rnn_state_actor"])] for j in range(512)]
exploration_timestamps, exploration_sequences, exploration_fish_positions = \
    extract_exploration_action_sequences_with_positions(
    data)

fig, ax = plt.subplots(figsize=(10, 5))
unit_29 = unit_activity1a[31]
unit_2 = unit_activity1a[45]

ax.plot(unit_29/max(unit_29))
ax.plot(unit_2/max(unit_2))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)

for i in exploration_timestamps:
    ax.hlines(-0.1, i[0], i[-1], color="g", linewidth=8)
plt.xlabel("Step", fontsize=25)
plt.ylabel("LSTM Unit Activity (Normalised)", fontsize=20)
plt.tight_layout()
plt.savefig("../../Figures/Panels/Panel-5/Exploration_unit.png")
plt.show()

# plt.eventplot(event_data, colors=color_set)
# for i in exploration_timestamps:
#     plt.hlines(9, i[0], i[-1])
#
# plt.show()
# display_neurons_with_selectivities("dqn_scaffold_14-1", False, "Prey-Static-15--0.3598551585021035", action_association=False)

# display_neurons_with_selectivities("dqn_scaffold_14-1", True, action_association=0)  # Neuron 31

# display_neurons_with_selectivities("dqn_scaffold_14-1", False, "Prey-Away--0.3598551585021035", action_association=3)
# display_neurons_with_selectivities("dqn_scaffold_14-1", False, "Prey-Away--1.2166531549356836", action_association=False)






