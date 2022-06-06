import numpy as np

from Analysis.load_data import load_data


def extract_consumption_action_sequences_with_positions(data, n=20):
    """Returns all action sequences that occur n steps before consumption, with behavioural choice and """
    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    prey_capture_timestamps_compiled = []
    fish_positions_compiled = []
    actions_compiled = []

    # Get the timestamps
    while len(consumption_timestamps) > 0:
        index = consumption_timestamps.pop(0)
        prey_capture_timestamps = [i for i in range(index-n+1, index+1) if i >= 0]
        prey_capture_timestamps_compiled.append(np.array(prey_capture_timestamps))

    # Get the positions and actions
    for consumption_sequence in prey_capture_timestamps_compiled:
        fish_positions = [data["fish_position"][i] for i in consumption_sequence]
        fish_positions_compiled.append(np.array(fish_positions))
        actions = [data["action"][i] for i in consumption_sequence]
        actions_compiled.append(np.array(actions))

    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]

    return prey_capture_timestamps_compiled, consumption_timestamps, actions_compiled, fish_positions_compiled


def extract_consumption_action_sequences(data, n=20):
    """Returns all action sequences that occur n steps before consumption"""
    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    prey_c_t = []
    action_sequences = []
    while len(consumption_timestamps) > 0:
        index = consumption_timestamps.pop(0)
        prey_capture_timestamps = [i for i in range(index-n+1, index+1) if i >= 0]
        prey_c_t.append(prey_capture_timestamps)
        action_sequence = [data["action"][i] for i in prey_capture_timestamps]
        action_sequences.append(action_sequence)
    return action_sequences, prey_c_t


def get_capture_sequences_with_energy_state(model_name, assay_config, assay_id, n):
    all_capture_sequences = []
    all_energy_states_by_sequence = []
    for i in range(1, n + 1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        new_energy_states = []
        new_sequences, new_timestamps = extract_consumption_action_sequences(data)
        for seq in new_timestamps:
            new_energy_states += [[data["energy_state"][i] for i in seq]]
        all_capture_sequences = all_capture_sequences + new_sequences
        all_energy_states_by_sequence += new_energy_states
    return all_capture_sequences, all_energy_states_by_sequence


def get_capture_sequences(model_name, assay_config, assay_id, n):
    all_capture_sequences = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_capture_sequences = all_capture_sequences + extract_consumption_action_sequences(data)[0]
    return all_capture_sequences
