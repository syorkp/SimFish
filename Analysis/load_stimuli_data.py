import json


def load_stimulus_data(model_name, assay_name):

    path = f"{model_name}/{assay_name}-stimuli_data.json"

    with open(f"../../Assay-Output/{path}") as file:
        data = json.load(file)

    stimuli_periods = []
    to_delete = []
    for i, entry in enumerate(data):
        if i == 0:
            for stimulus in entry.keys():
                data[i][stimulus]["Pre-onset"] = 0
        else:
            if i % 2 == 0:
                for stimulus in entry.keys():
                    data[i][stimulus]["Pre-onset"] = data[i-1][stimulus]["Pre-onset"]
            else:
                to_delete.append(i)

    for i in reversed(to_delete):
        del data[i]

    return data


# format this data to be useful for comparison.

# load_stimulus_data("changed_penalties-1", "Controlled_Visual_Stimuli")



