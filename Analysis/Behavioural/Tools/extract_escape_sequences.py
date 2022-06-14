import numpy as np


def extract_escape_action_sequences_with_positions(data):
    predator_timestamps = [i for i, a in enumerate(data["predator_presence"]) if a == 1]
    avoidance_timestamps_compiled = []
    fish_positions_compiled = []
    actions_compiled = []

    # Convert timestamps into sequences
    current_sequence = []
    for i, t in enumerate(predator_timestamps):
        if i == 0:
            current_sequence.append(t)
        else:
            if t-1 == predator_timestamps[i-1]:
                current_sequence.append(t)
            else:
                avoidance_timestamps_compiled.append(current_sequence)
                current_sequence = []
                current_sequence.append(t)
    avoidance_timestamps_compiled.append(current_sequence)

    # Get the positions and actions
    for avoidance_sequence in avoidance_timestamps_compiled:
        fish_positions = [data["fish_position"][i] for i in avoidance_sequence]
        fish_positions_compiled.append(np.array(fish_positions))
        actions = [data["action"][i] for i in avoidance_sequence]
        actions_compiled.append(np.array(actions))

    return avoidance_timestamps_compiled, actions_compiled, fish_positions_compiled








