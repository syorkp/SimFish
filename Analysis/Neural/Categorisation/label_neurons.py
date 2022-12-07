import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd


def check_boring_unit(vector):
    if all(r <= 0 for r in vector):# and abs(max(vector)) - abs(min(vector)) < abs(np.mean(vector)):
        return "Neither"
    elif all(r >= 0 for r in vector): # and abs(max(vector)) - abs(min(vector)) < abs(np.mean(vector)):
        return "Neither"
    else:
        return None


def simple_check_feature_selectivity(vector, selectivity, threshold=0.5):
    max_resp = max(vector)
    min_resp = min(vector)
    if max_resp > threshold or min_resp < -threshold:
        return selectivity
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


def label_all_units_selectivities(response_vectors, stimulus_vectors, check_predator=False, background=False):
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
        prey = simple_check_feature_selectivity(prey_subset, "Prey")
        if prey is not None:
            selectivities[i]["Prey"] = []
            for j, stim in enumerate(stimulus_vectors_subset):
                sel = simple_check_feature_selectivity([prey_subset[j]], stim)
                if sel is not None:
                    selectivities[i]["Prey"].append(sel)

        predator_subset = unit[121:242]
        stimulus_vectors_subset = stimulus_vectors[121:242]
        if check_predator:
            predator = simple_check_feature_selectivity(predator_subset, "Predator")
            if predator is not None:
                selectivities[i]["Predator"] = []
                for j, stim in enumerate(stimulus_vectors_subset):
                    sel = simple_check_feature_selectivity([predator_subset[j]], stim)
                    if sel is not None:
                        selectivities[i]["Predator"].append(sel)

        if background:
            red_prey_subset = unit[242:363]
            stimulus_vectors_subset = stimulus_vectors[242:363]
            red_prey = simple_check_feature_selectivity(red_prey_subset, "Prey")
            if red_prey is not None:
                selectivities[i]["Background-Prey"] = []
                for j, stim in enumerate(stimulus_vectors_subset):
                    sel = simple_check_feature_selectivity([red_prey_subset[j]], stim)
                    if sel is not None:
                        selectivities[i]["Background-Prey"].append(sel)

            red_predator_subset = unit[363:]
            stimulus_vectors_subset = stimulus_vectors[363:]
            red_predator = simple_check_feature_selectivity(red_predator_subset, "Predator")
            if red_predator is not None:
                selectivities[i]["Background-Predator"] = []
                for j, stim in enumerate(stimulus_vectors_subset):
                    sel = simple_check_feature_selectivity([red_predator_subset[j]], stim)
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


def get_with_selectivity(features, selectivities):
    ns = []
    for n, neuron in enumerate(selectivities):
        for key in neuron.keys():
            for s in neuron[key]:
                if any(feature in s for feature in features):
                    ns.append(n)
    return list(set(ns))


def display_categories_counts(category_list):
    pap = []
    pred = []
    prey = []
    neither = []
    for categories in category_list:
        pap.append(categories.count("Prey-and-Predator"))
        pred.append(categories.count("Predator-Only"))
        prey.append(categories.count("Prey-Only"))
        neither.append(categories.count("Neither"))
    data = pd.DataFrame([pap, pred, prey, neither])
    sns.set()
    df = pd.DataFrame(data).T
    df = df.rename(columns={k: f'Data{k + 1}' for k in range(len(data))}).reset_index()
    df = pd.wide_to_long(df, stubnames=['Data'], i='index', j='ID').reset_index()[['ID', 'Data']]
    plt.figure(figsize=(7, 6))
    ax = sns.boxplot(x='ID', y='Data', data=df, fliersize=0)
    ax = sns.stripplot(y="Data", x="ID", data=df, color=".3")
    plt.xticks([0, 1, 2, 3], ["Prey-and-Predator", "Predator", "Prey", "Neither"])
    plt.xlabel("Ethological categories", fontsize=15)
    plt.ylabel("Number of Neurons", fontsize=15)
    plt.show()


def display_selectivity_tally2(selectivities_list):
    model_counts = []
    for selectivities in selectivities_list:
        counts = []
        for i, neuron in enumerate(selectivities):
            new_sel = []
            for key in neuron.keys():
                if type(neuron[key]) is list:
                    new_sel = new_sel + neuron[key]
                else:
                    new_sel.append(neuron[key])
            for j, sel in enumerate(new_sel):
                sel = sel.split("-")[:-1]
                n = ""
                for cc in sel:
                    n = n + "-" + cc
                if n != "":
                    new_sel[j] = n
            for i, count in enumerate(new_sel):
                if count[0] == "-":
                    new_sel[i] = count[1:]
                if count[-1] == "-":
                    new_sel[i] = count[:-1]
            for i, count in enumerate(new_sel):
                if count[0] == "-":
                    new_sel[i] = count[1:]
                if count[-1] == "-":
                    new_sel[i] = count[:-1]
            new_sel = list(set(new_sel))
            counts = counts + new_sel
        model_counts.append(counts)

    sl = ['Prey-Static-5', 'Prey-Static-10', 'Prey-Static-15','Prey-Left-5','Prey-Left-10','Prey-Left-15',
           'Prey-Right-5', 'Prey-Right-10','Prey-Right-15','Prey-Towards','Prey-Away','Predator-Static-40',
           'Predator-Static-60',  'Predator-Static-80','Predator-Left-40', 'Predator-Left-60',   'Predator-Left-80',
           'Predator-Right-40',  'Predator-Right-60',  'Predator-Right-80', 'Predator-Towards', 'Predator-Away', 'Neither', ]
    selectivity_counts_for_models = []
    for stim in sl:
        selectivity_counts_for_models.append([model_counts[i].count(stim) for i in range(len(model_counts))])
    means = []
    sds = []
    for s in selectivity_counts_for_models:
        means.append(np.mean(s))
        sds.append(np.std(s))


    ticks = ['Static-5', 'Static-10', 'Static-15','Left-5','Left-10','Left-15',
           'Right-5', 'Right-10','Right-15','Prey-Towards','Prey-Away','Static-40',
           'Static-60',  'Static-80','Left-40', 'Left-60',   'Left-80',
           'Right-40',  'Right-60',  'Right-80', 'Predator-Towards', 'Predator-Away', 'None', ]

    sns.set()
    fig, ax = plt.subplots()
    ax.bar(ticks, means, yerr=sds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticklabels(ticks, fontsize=13)
    ax.yaxis.grid(True)
    plt.xlabel("Stimulus", fontsize=30)
    plt.ylabel("Neurons Responsive", fontsize=25)
    fig.set_size_inches((14, 8))
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.show()


def display_selectivity_tally(selectivities, stimulus_vector):
    counts = []
    for i, neuron in enumerate(selectivities):
        new_sel = []
        for key in neuron.keys():
            if type(neuron[key]) is list:
                new_sel = new_sel + neuron[key]
            else:
                new_sel.append(neuron[key])
        for j, sel in enumerate(new_sel):
            sel = sel.split("-")[:-1]
            n = ""
            for cc in sel:
                n = n + "-" + cc
            if n != "":
                new_sel[j] = n
        for i, count in enumerate(new_sel):
            if count[0] == "-":
                new_sel[i] = count[1:]
            if count[-1] == "-":
                new_sel[i] = count[:-1]
        for i, count in enumerate(new_sel):
            if count[0] == "-":
                new_sel[i] = count[1:]
            if count[-1] == "-":
                new_sel[i] = count[:-1]
        new_sel = list(set(new_sel))
        counts = counts + new_sel

    fig, ax = plt.subplots(figsize=(15,8))

    sl = ['Prey-Static-5', 'Prey-Static-10', 'Prey-Static-15','Prey-Left-5','Prey-Left-10','Prey-Left-15',
           'Prey-Right-5', 'Prey-Right-10','Prey-Right-15','Prey-Towards','Prey-Away','Predator-Static-40',
           'Predator-Static-60',  'Predator-Static-80','Predator-Left-40', 'Predator-Left-60',   'Predator-Left-80',
           'Predator-Right-40',  'Predator-Right-60',  'Predator-Right-80', 'Predator-Towards', 'Predator-Away', 'Neither', ]
    ticks = ['Static-5', 'Static-10', 'Static-15','Left-5','Left-10','Left-15',
           'Right-5', 'Right-10','Right-15','Prey-Towards','Prey-Away','Static-40',
           'Static-60',  'Static-80','Left-40', 'Left-60',   'Left-80',
           'Right-40',  'Right-60',  'Right-80', 'Predator-Towards', 'Predator-Away', 'None', ]
    g = sns.countplot(counts, color="blue", order=sl)
    plt.xlabel("Stimulus", fontsize=30)
    plt.ylabel("Neurons Selective", fontsize=30)
    plt.xticks(rotation=90)
    ax.set_xticklabels(ticks)
    fig.tight_layout()

    plt.show()

# cats = []
# sels = []
# for i in [4, 5, 6, 8]:
#     full_rv = create_full_response_vector(f"new_even_prey_ref-{i}")
#     full_sv = create_full_stimulus_vector(f"new_even_prey_ref-{i}")
#     sel = label_all_units_selectivities(full_rv, full_sv)
#     sels.append(sel)
#     # display_selectivity_tally(sel, full_sv)
#     cat = assign_neuron_names(sel)
#     predator_ns = [j for j, c in enumerate(cat) if c == "Prey-Only"]
#     cats.append(cat)
# display_selectivity_tally2(sels)
# display_categories_counts(cats)
#
# cats = []

# for i in [3, 4, 5, 6]:
#     full_rv = create_full_response_vector(f"new_differential_prey_ref-{i}")
#     full_sv = create_full_stimulus_vector(f"new_differential_prey_ref-{i}")
#     sel = label_all_units_selectivities(full_rv, full_sv)
#     # display_selectivity_tally(sel, full_sv)
#     cat = assign_neuron_names(sel)
#     cats.append(cat)
# display_categories_counts(cats)

# x = True
# # neurons_to_ablate = get_with_selectivity(["Prey-Static-15", "Prey-Left-15", "Prey-Right-15"], sel)
# # neurons_to_ablate = get_with_selectivity(["Prey-Static-5-0.07", "Prey-Left-5-0.07", "Prey-Right-5-0.07",
# #                                           "Prey-Static-10-0.07", "Prey-Left-10-0.07", "Prey-Right-10-0.07",
# #                                           "Prey-Static-15-0.07", "Prey-Left-15-0.07", "Prey-Right-15-0.07"], sel)
# neurons_to_ablate = get_with_selectivity(["Prey-Static-15-0.07", "Prey-Left-15-0.07", "Prey-Right-15-0.07"], sel)
# print(neurons_to_ablate)

