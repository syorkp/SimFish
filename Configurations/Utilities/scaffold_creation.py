import json
import os


def save_files(scaffold_name, env, params, n):
    """Saves scaffold files at a given point."""

    with open(f"Configurations/Training-Configs/{scaffold_name}/{str(n)}_env.json", 'w') as f:
        json.dump(env, f, indent=4)

    with open(f"Configurations/Training-Configs/{scaffold_name}/{str(n)}_learning.json", 'w') as f:
        json.dump(params, f, indent=4)


def save_transitions(scaffold_name, transitions):
    with open(f"Configurations/Training-Configs/{scaffold_name}/transitions.json", 'w') as f:
        json.dump(transitions, f, indent=4)


def create_scaffold(scaffold_name, initial_env, initial_params, changes):
    """
    Given complete list of changes, and original scaffold, creates the necessary scaffold files.

    :param scaffold_name:
    :param initial_env:
    :param initial_params:
    :param changes: A list of lists, with dims (n_configs x 3), with columns being:
        - threshold measure (str),
        - threshold (float),
        - change to make (str)
        - new value (unspecified)
    :return:
    """
    if not os.path.exists(f"Configurations/Training-Configs/{scaffold_name}/"):
        os.makedirs(f"Configurations/Training-Configs/{scaffold_name}/")

    transitions = {
        "Episode": {},
        "PCI": {},
        "PAI": {},
        "SGB": {},
    }
    save_files(scaffold_name, initial_env, initial_params, 1)

    for i, change in enumerate(changes):
        if len(change) > 4:
            initial_params[change[2]] = change[3]
        else:
            initial_env[change[2]] = change[3]
        save_files(scaffold_name, initial_env, initial_params, i+2)

        transitions[change[0]][str(i + 2)] = change[1]
    save_transitions(scaffold_name, transitions)


def build_changes_list_gradual(threshold_measure, threshold, change_to_make, initial_value, final_value, num_steps):
    """Tool to output gradual changes in a parameter.
    Needs to be a generator???
    """
    increment = (final_value-initial_value)/num_steps
    current_value = initial_value
    changes = []
    for step in range(num_steps):
        current_value += increment
        changes += [[threshold_measure, threshold, change_to_make, current_value]]
    return changes
