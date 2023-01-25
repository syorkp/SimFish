import json
import os


def save_files(scaffold_name, env, params, n):
    """Saves scaffold files at a given point."""

    with open(f"Configurations/Training-Configs/{scaffold_name}/{str(n)}_env.json", 'w') as f:
        json.dump(env, f, indent=4)

    with open(f"Configurations/Training-Configs/{scaffold_name}/{str(n)}_learning.json", 'w') as f:
        json.dump(params, f, indent=4)


def save_transitions(scaffold_name, transitions, finished_condition):
    transitions["Finished Condition"] = finished_condition
    with open(f"Configurations/Training-Configs/{scaffold_name}/transitions.json", 'w') as f:
        json.dump(transitions, f, indent=4)


def create_transitions_log(scaffold_name, changes):
    with open(f"Configurations/Training-Configs/{scaffold_name}/all_transitions.txt", "w") as f:
        for i, change in enumerate(changes):
            if len(change) > 4:
                f.write(f"{str(i + 1)}) Metric-{change[0]}: {change[1]}, Multiple changes")
            else:
                f.write(f"{str(i + 1)}) Metric-{change[0]}: {change[1]}, Variable-{change[2]}: {change[3]}")
            f.write('\n')


def create_scaffold(scaffold_name, initial_env, initial_params, changes, finished_condition):
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
    create_transitions_log(scaffold_name, changes)

    for i, change in enumerate(changes):
        if change[-1] == "do_to_params":
            # Changes not to environment, but to learning_params
            if len(change) > 5:
                things_to_change = change[2:-1]
                for j in range(0, len(things_to_change), 2):
                    initial_params[things_to_change[j]] = things_to_change[j + 1]
            else:
                initial_params[change[2]] = change[3]
        elif change[-1] == "complex":
            # To allow multiple changes to both env and params
            initial_env, initial_params = implement_complex_transitions(change[2:-1], initial_env, initial_params)
        else:
            # To allow mulitple changes
            if len(change) > 4:
                # Implement multiple changes
                things_to_change = change[2:]
                for j in range(0, len(things_to_change), 2):
                    initial_env[things_to_change[j]] = things_to_change[j + 1]
            else:
                initial_env[change[2]] = change[3]


        save_files(scaffold_name, initial_env, initial_params, i + 2)

        transitions[change[0]][str(i + 2)] = change[1]
    save_transitions(scaffold_name, transitions, finished_condition)


def build_changes_list_gradual(threshold_measure, threshold, change_to_make, initial_value, final_value, num_steps,
                               discrete):
    """Tool to output gradual changes in a parameter.
    Needs to be a generator???
    """
    increment = (final_value - initial_value) / num_steps
    current_value = initial_value
    changes = []
    for step in range(num_steps):
        current_value += increment
        if discrete:
            current_value = int(current_value)
        changes += [[threshold_measure, threshold, change_to_make, current_value]]
    return changes


def implement_complex_transitions(list_of_changes, initial_env, initial_params):
    """Method that returns the desired env and params from list of changes to both env and params."""
    for i in range(0, len(list_of_changes), 2):
        if list_of_changes[i] in initial_env:
            initial_env[list_of_changes[i]] = list_of_changes[i+1]
        elif list_of_changes[i] in initial_params:
            initial_params[list_of_changes[i]] = list_of_changes[i + 1]
        else:
            print(f"Error, {list_of_changes[i]} does not exist in config specification")
    return initial_env, initial_params
