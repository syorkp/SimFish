import numpy as np

from Analysis.load_data import load_data


def extract_capture_sequences_with_additional_steps(data, n=20, n2=3):
    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    prey_capture_timestamps_compiled = []
    fish_positions_compiled = []
    actions_compiled = []
    prey_positions_compiled = []

    # Get the timestamps
    while len(consumption_timestamps) > 0:
        index = consumption_timestamps.pop(0)
        prey_capture_timestamps = [i for i in range(index-n+1, index+1+n2) if i >= 0]
        prey_capture_timestamps_compiled.append(np.array(prey_capture_timestamps))

    # Get the positions and actions
    for consumption_sequence in prey_capture_timestamps_compiled:
        fish_positions = [data["fish_position"][i] for i in consumption_sequence]
        fish_positions_compiled.append(np.array(fish_positions))
        actions = [data["action"][i] for i in consumption_sequence]
        actions_compiled.append(np.array(actions))
        prey_positions = [data["prey_positions"][i] for i in consumption_sequence]
        prey_positions_compiled.append(np.array(prey_positions))

    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]

    return prey_capture_timestamps_compiled, consumption_timestamps, actions_compiled, fish_positions_compiled, prey_positions_compiled


def remove_sequences_with_nearby_prey_following_capture(actions_compiled, fish_positions_compiled, prey_positions_compiled, visual_range=100, n2=3):

    for i, sequence in enumerate(actions_compiled):
        fish_prey_vectors = prey_positions_compiled[i][-n2:] - np.expand_dims(fish_positions_compiled[i][-n2:], 1)
        fish_prey_distances = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
        prey_within_range = (fish_prey_distances < visual_range)
        if np.sum(prey_within_range) > 0:
            pass
        else:
            ... # TODO: Add to a buffer.
        x = True

data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic-1")
prey_capture_timestamps_compiled, consumption_timestamps, actions_compiled, fish_positions_compiled, prey_positions_compiled = extract_capture_sequences_with_additional_steps(data)
remove_sequences_with_nearby_prey_following_capture(actions_compiled, fish_positions_compiled, prey_positions_compiled)

# Need to extract capture sequences, but keep 3 points afterwards.
# Remove those where prey were nearby following initial capture.