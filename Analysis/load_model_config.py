import json


def load_configuration_files(model_name):
    """Loads the data of an individual assay from an assay configuration file."""
    try:
        model_location = f"../../../Training-Output/{model_name}"

        with open(f"{model_location}/learning_configuration.json", 'r') as f:
            params = json.load(f)
        with open(f"{model_location}/environment_configuration.json", 'r') as f:
            env = json.load(f)
    except FileNotFoundError:
        model_location = f"../../Training-Output/{model_name}"

        with open(f"{model_location}/learning_configuration.json", 'r') as f:
            params = json.load(f)
        with open(f"{model_location}/environment_configuration.json", 'r') as f:
            env = json.load(f)


    base_network_layers = params["base_network_layers"]
    ops = params["ops"]
    connectivity = params["connectivity"]
    return params, env, base_network_layers, ops, connectivity

