

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
        prey_capture_timestamps_compiled.append(prey_capture_timestamps)

    # Get the positions and actions
    for consumption_sequence in prey_capture_timestamps_compiled:
        fish_positions = [data["fish_position"][i] for i in consumption_sequence]
        fish_positions_compiled.append(fish_positions)
        actions = [data["action"][i] for i in consumption_sequence]
        actions_compiled.append(actions)

    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]

    return prey_capture_timestamps_compiled, consumption_timestamps, actions_compiled, fish_positions_compiled
