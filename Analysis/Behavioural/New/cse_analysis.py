from Analysis.load_data import load_data
from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.Behavioural.bout_transition_probabilities import get_third_order_transition_counts_from_sequences, get_modal_sequences, compute_transition_probabilities
from Analysis.Behavioural.display_action_sequences import display_sequences
from Analysis.Neural.calculate_vrv import create_full_stimulus_vector


def get_behavioural_vector(all_data, stimulus, stimulus_data):
    vector = []
    if stimulus_data[0][list(stimulus_data[0].keys())[0]]['Angle'] > 0:
        reverse = True
    else:
        reverse = False

    for period in stimulus_data:
        interval = period[stimulus]["Onset"] - period[stimulus]["Pre-onset"]
        vector.append(all_data["behavioural choice"][period[stimulus]["Onset"]: period[stimulus]["Onset"] + interval])

    if reverse:
        vector.reverse()
    return vector


def create_behavioural_response_vector(model_name, background=False):
    # Creates the full 484 dimensional response vector.
    action_vector = []
    if background:
        file_precursors = ["Prey", "Predator", "Background-Prey", "Background-Predator"]
    else:
        file_precursors = ["Prey", "Predator"]
    prey_assay_ids = ["Prey-Static-5", "Prey-Static-10", "Prey-Static-15",
                      "Prey-Left-5", "Prey-Left-10", "Prey-Left-15",
                      "Prey-Right-5", "Prey-Right-10", "Prey-Right-15",
                      "Prey-Away", "Prey-Towards"]
    predator_assay_ids = ["Predator-Static-40", "Predator-Static-60", "Predator-Static-80",
                          "Predator-Left-40", "Predator-Left-60", "Predator-Left-80",
                          "Predator-Right-40", "Predator-Right-60", "Predator-Right-80",
                          "Predator-Away", "Predator-Towards"]
    for file_p in file_precursors:
        if "Prey" in file_p:
            for aid in prey_assay_ids:
                data = load_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                stimulus_data = load_stimulus_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                new_vector_section = get_behavioural_vector(data, "prey 1", stimulus_data)
                action_vector = action_vector + new_vector_section
        elif "Predator" in file_p:
            for aid in predator_assay_ids:
                data = load_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                stimulus_data = load_stimulus_data(model_name, f"{file_p}-Full-Response-Vector", aid)
                new_vector_section = get_behavioural_vector(data, "predator 1", stimulus_data)
                action_vector = action_vector + new_vector_section
    return action_vector


full_rv = create_behavioural_response_vector("even_prey_ref-5")
full_sv = create_full_stimulus_vector("even_prey_ref-5")
modal_sequence_v = []

for seq in full_rv:
    tc = get_third_order_transition_counts_from_sequences([seq])
    tp = compute_transition_probabilities(tc)
    ms = get_modal_sequences(tp, order=3, number=1)
    modal_sequence_v.append(ms[0])

display_sequences(modal_sequence_v)
x = True