import json


def get_scaffold_num_points(config_name):
    try:
        model_location = f"../../../Configurations/Training-Configs/{config_name}/all_transitions.txt"

        with open(f"{model_location}", 'r') as f:
            all_transitions = f.readlines()

    except FileNotFoundError:
        try:
            model_location = f"../../Configurations/Training-Configs/{config_name}/all_transitions.txt"

            with open(f"{model_location}", 'r') as f:
                all_transitions = f.readlines()
        except FileNotFoundError:
            try:
                model_location = f"../Configurations/Training-Configs/{config_name}/all_transitions.txt"

                with open(f"{model_location}", 'r') as f:
                    all_transitions = f.readlines()
            except FileNotFoundError:
                model_location = f"Configurations/Training-Configs/{config_name}/all_transitions.txt"

                with open(f"{model_location}", 'r') as f:
                    all_transitions = f.readlines()

    num_scaffold_points = len(all_transitions) + 1
    return num_scaffold_points


def load_configuration_files_by_scaffold_point(config_name, scaffold_point):
    try:
        model_location = f"../../../Configurations/Training-Configs/{config_name}/{scaffold_point}"

        with open(f"{model_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{model_location}_env.json", 'r') as f:
            env = json.load(f)
    except FileNotFoundError:
        try:
            model_location = f"../../Configurations/Training-Configs/{config_name}/{scaffold_point}"

            with open(f"{model_location}_learning.json", 'r') as f:
                params = json.load(f)
            with open(f"{model_location}_env.json", 'r') as f:
                env = json.load(f)
        except FileNotFoundError:
            try:
                model_location = f"../Configurations/Training-Configs/{config_name}/{scaffold_point}"

                with open(f"{model_location}_learning.json", 'r') as f:
                    params = json.load(f)
                with open(f"{model_location}_env.json", 'r') as f:
                    env = json.load(f)
            except FileNotFoundError:
                model_location = f"Configurations/Training-Configs/{config_name}/{scaffold_point}"

                with open(f"{model_location}_learning.json", 'r') as f:
                    params = json.load(f)
                with open(f"{model_location}_env.json", 'r') as f:
                    env = json.load(f)

    return env, params


def load_assay_configuration_files(model_name):
    """Loads the data of an individual assay from an assay configuration file."""
    try:
        model_location = f"../../../Training-Output/{model_name}"

        with open(f"{model_location}/learning_configuration.json", 'r') as f:
            params = json.load(f)
        with open(f"{model_location}/environment_configuration.json", 'r') as f:
            env = json.load(f)
    except FileNotFoundError:
        try:
            model_location = f"../../Training-Output/{model_name}"

            with open(f"{model_location}/learning_configuration.json", 'r') as f:
                params = json.load(f)
            with open(f"{model_location}/environment_configuration.json", 'r') as f:
                env = json.load(f)
        except FileNotFoundError:
            try:
                model_location = f"../Training-Output/{model_name}"

                with open(f"{model_location}/learning_configuration.json", 'r') as f:
                    params = json.load(f)
                with open(f"{model_location}/environment_configuration.json", 'r') as f:
                    env = json.load(f)
            except FileNotFoundError:
                model_location = f"Training-Output/{model_name}"

                with open(f"{model_location}/learning_configuration.json", 'r') as f:
                    params = json.load(f)
                with open(f"{model_location}/environment_configuration.json", 'r') as f:
                    env = json.load(f)

    base_network_layers = params["base_network_layers"]
    ops = params["ops"]
    connectivity = params["connectivity"]
    return params, env, base_network_layers, ops, connectivity


if __name__ == "__main__":

    get_scaffold_num_points("dqn_beta")
