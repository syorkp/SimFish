import json

from Analysis.load_data import load_data
from Analysis.Neural.event_triggered_averages import get_eta, get_for_specific_neurons
from Analysis.Neural.label_neurons import label_all_units_selectivities, get_with_selectivity, assign_neuron_names
from Analysis.Neural.calculate_vrv import create_full_response_vector, create_full_stimulus_vector


def get_formatted_data(broad_categories, atas, sel, threshold=10):
    data = {}
    for neuron in range(len(broad_categories)):
        selectivities = []
        for key in sel[neuron].keys():
            selec = sel[neuron][key]
            if type(selec) is not list:
                selec = [selec]
            selectivities = selectivities + selec
        action_associations = {str(a): atas[str(a)][neuron] for a in range(10) if atas[str(a)][neuron]>threshold}
        data[f"Neuron {neuron}"] = {"category":broad_categories[neuron],
                                    "selectivities": selectivities,
                                    "action_associations": action_associations}
    return data


def save_all_categorisations(model_name):
    event_triggered_average_data = load_data(model_name, "Naturalistic", "Naturalistic-1")
    action_triggered_averages = get_eta(event_triggered_average_data, "actions")
    # TODO: Make work for any other associations
    full_rv = create_full_response_vector(model_name)
    full_sv = create_full_stimulus_vector(model_name)
    sel = label_all_units_selectivities(full_rv, full_sv)
    broad_categories = assign_neuron_names(sel)
    d = get_formatted_data(broad_categories, action_triggered_averages, sel)
    with open(f"../Categorisation-Data/{model_name}_categories.json", "w") as outfile:
        json.dump(d, outfile, indent=4)


save_all_categorisations("even_prey_ref-4")
save_all_categorisations("even_prey_ref-5")
save_all_categorisations("even_prey_ref-6")
save_all_categorisations("even_prey_ref-7")
