import json


def new_load_stimulus_data(model_name, assay_name, assay_id):
    path = f"{model_name}/{assay_name}-{assay_id}-stimuli_data.json"

    with open(f"../../../Assay-Output/{path}") as file:
        data = json.load(file)

    to_delete = []
    for i, entry in enumerate(data):
        if i < 2:
            continue
        if i % 3 == 2:
            for stimulus in entry.keys():
                data[i][stimulus]["Initialisation"] = data[i - 2][stimulus]["Initialisation"]
                data[i][stimulus]["Pre-onset"] = data[i - 1][stimulus]["Pre-onset"]
        else:
            to_delete.append(i)

    for i in reversed(to_delete):
        del data[i]

    return data


def load_stimulus_data(model_name, assay_name, assay_id):
    path = f"{model_name}/{assay_name}-{assay_id}-stimuli_data.json"

    with open(f"../../../Assay-Output/{path}") as file:
        data = json.load(file)

    to_delete = []
    for i, entry in enumerate(data):
        if i == 0:
            for stimulus in entry.keys():
                data[i + 1][stimulus]["Initialisation"] = 0
                data[i + 1][stimulus]["Pre-onset"] = data[i][stimulus]["Pre-onset"]
            to_delete.append(0)
        elif i == 1:
            pass
        elif 1 < i <= 3:
            to_delete.append(i)
        else:
            if i % 3 == 1:
                for stimulus in entry.keys():
                    data[i][stimulus]["Initialisation"] = data[i - 2][stimulus]["Initialisation"]
                    data[i][stimulus]["Pre-onset"] = data[i - 1][stimulus]["Pre-onset"]
            else:
                to_delete.append(i)

    for i in reversed(to_delete):
        del data[i]

    return data

# format this data to be useful for comparison.

# load_stimulus_data("changed_penalties-1", "Controlled_Visual_Stimuli")
