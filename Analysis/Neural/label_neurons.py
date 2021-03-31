import numpy as np


from Analysis.Neural.calculate_vrv import create_full_response_vector, create_full_stimulus_vector


def check_boring_unit(vector):
    if all(r <= 0 for r in vector):# and abs(max(vector)) - abs(min(vector)) < abs(np.mean(vector)):
        return "Unvexed Up"
    elif all(r >= 0 for r in vector): # and abs(max(vector)) - abs(min(vector)) < abs(np.mean(vector)):
        return "Unvexed Down"
    else:
        return None


def check_feature_selectivity(vector, selectivity, threshold=0.5):
    """Check responsiveness to prey or predator."""
    max_resp = max(vector)
    min_resp = min(vector)
    if max_resp > threshold or min_resp < -threshold:
        if max_resp > 0.9:
            selectivity += " - Strong Positive"
        elif max_resp > threshold:
            selectivity += " - Positive"
        if min_resp < -0.9:
            selectivity += " - Strong Negative"
        elif min_resp < -threshold:
            selectivity += " - Negative"
        return selectivity
    else:
        return None


def normalise_response_vectors(response_vectors):
    # normalise within each vector
    response_vectors = np.array(response_vectors)
    for i, v in enumerate(response_vectors):
        m = max([abs(min(v)), abs(max(v))])
        response_vectors[i, :] = np.interp(response_vectors[i, :], (-m, m), (-1, 1))

    # normalise across vectors NOTE: May result in high selectivity for features none are selective for.
    for i, v in enumerate(response_vectors[0]):
        m = max([abs(min(response_vectors[:, i])), abs(max(response_vectors[:, i]))])

        response_vectors[:, i] = np.interp(response_vectors[:, i], (-m, m), (-1, 1))
    return response_vectors


def label_all_units_selectivities(response_vectors, stimulus_vectors, background=False):
    """Returns a neuron dimensional list of lists, each of which contains all selectivity properties of neurons.
    Higher level properties could then be inferred by combinations of selectivities"""

    response_vectors = normalise_response_vectors(response_vectors)
    selectivities = [{} for i in response_vectors]
    for i, unit in enumerate(response_vectors):
        uv = check_boring_unit(unit)
        if uv:
            selectivities[i]["Boring"] = uv
            continue
        prey_subset = unit[:121]
        stimulus_vectors_subset = stimulus_vectors[:121]
        prey = check_feature_selectivity(prey_subset, "Prey")
        if prey is not None:
            selectivities[i]["Prey"] = []
            for j, stim in enumerate(stimulus_vectors_subset):
                sel = check_feature_selectivity([prey_subset[j]], stim)
                if sel is not None:
                    selectivities[i]["Prey"].append(sel)

        predator_subset = unit[121:242]
        stimulus_vectors_subset = stimulus_vectors[121:242]
        predator = check_feature_selectivity(predator_subset, "Predator")
        if predator is not None:
            selectivities[i]["Predator"] = []
            for j, stim in enumerate(stimulus_vectors_subset):
                sel = check_feature_selectivity([predator_subset[j]], stim)
                if sel is not None:
                    selectivities[i]["Predator"].append(sel)

        if background:
            red_prey_subset = unit[242:363]
            stimulus_vectors_subset = stimulus_vectors[242:363]
            red_prey = check_feature_selectivity(red_prey_subset, "Prey")
            if red_prey is not None:
                selectivities[i]["Background-Prey"] = []
                for j, stim in enumerate(stimulus_vectors_subset):
                    sel = check_feature_selectivity([red_prey_subset[j]], stim)
                    if sel is not None:
                        selectivities[i]["Background-Prey"].append(sel)

            red_predator_subset = unit[363:]
            stimulus_vectors_subset = stimulus_vectors[363:]
            red_predator = check_feature_selectivity(red_predator_subset, "Predator")
            if red_predator is not None:
                selectivities[i]["Background-Predator"] = []
                for j, stim in enumerate(stimulus_vectors_subset):
                    sel = check_feature_selectivity([red_predator_subset[j]], stim)
                    if sel is not None:
                        selectivities[i]["Background-Predator"].append(sel)
    return selectivities


def assign_neuron_names(selectivities):
    categories = []
    for unit in selectivities:
        keys = unit.keys()
        if "Boring" in keys:
            categories.append(unit["Boring"])
        elif "Prey" in keys and "Predator" not in keys:
            categories.append("Prey-Only")
        elif "Predator" in keys and "Prey" not in keys:
            categories.append("Predator-Only")
        elif "Prey" in keys and "Predator" in keys:
            categories.append("Prey-and-Predator")
        else:
            print("Error")
            categories.append("None")
    return categories


def label_all_units_tsne(response_vectors, archetypes):
    """Based on the archetypes, clusters the nearest units and assigns this manual category to them."""
    ...


full_rv = create_full_response_vector("even_prey_ref-5")
full_sv = create_full_stimulus_vector("even_prey_ref-5")
sel = label_all_units_selectivities(full_rv, full_sv)
cat = assign_neuron_names(sel)

archetypes = [[5, "Prey"],
              [10, "Predator"]]

x = True
