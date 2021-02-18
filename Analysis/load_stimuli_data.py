import json


path = "changed_penalties-1/Controlled_Visual_Stimuli-stimuli_data.json"

with open(f"../Assay-Output/{path}") as file:
    data = json.load(file)

x = True



