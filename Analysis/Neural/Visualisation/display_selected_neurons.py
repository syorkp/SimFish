import json

from Analysis.Neural.Visualisation.plot_activity import plot_activity


from Analysis.Neural.Visualisation.display_many_neurons import plot_traces
from Analysis.load_data import load_data
from Analysis.load_stimuli_data import load_stimulus_data


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
    rnn_data = data["rnn_state"][:, 0, 0, :]

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


# display_neurons_with_selectivities("dqn_scaffold_14-1", False, "Prey-Static-15--0.3598551585021035", action_association=False)
# display_neurons_with_selectivities("dqn_scaffold_14-1", True, action_association=0)  # Neuron 31
# display_neurons_with_selectivities("dqn_scaffold_14-1", False, "Prey-Away--0.3598551585021035", action_association=3)
# display_neurons_with_selectivities("dqn_scaffold_14-1", False, "Prey-Away--1.2166531549356836", action_association=False)






