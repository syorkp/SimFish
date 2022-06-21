import json

model_name = "dqn_scaffold_14-1"
config_name = "Prey-Full-Response-Vector"
correct_assay_id = "Prey-Static-5"
all_other_assay_id = ["Prey-Static-10", "Prey-Static-15",
                      "Prey-Left-5", "Prey-Left-10", "Prey-Left-15",
                      "Prey-Right-5", "Prey-Right-10", "Prey-Right-15",
                      "Prey-Away", "Prey-Towards"]

with open(f"../../Assay-Output/{model_name}/{config_name}-{correct_assay_id}-stimuli_data.json", "r") as file:
    correct_contents = json.load(file)

for aid in all_other_assay_id:
    with open(f"../../Assay-Output/{model_name}/{config_name}-{aid}-stimuli_data.json", "w") as file:
        json.dump(correct_contents, file)

