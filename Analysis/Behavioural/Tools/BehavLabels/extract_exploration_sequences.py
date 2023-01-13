import numpy as np

from Analysis.load_data import load_data


def get_free_swimming_sequences(data):
    """OLD: Requires the following data: position, prey_positions, predator. Assumes square arena 1500."""
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]
    wall_timestamps = [i for i, p in enumerate(data["position"]) if 200 < p[0] < 1300 and 200<p[1]<1300]
    prey_timestamps = []
    sensing_distance = 200
    for i, p in enumerate(data["position"]):
        for prey in data["prey_positions"][i]:
            sensing_area = [[p[0] - sensing_distance,
                             p[0] + sensing_distance],
                            [p[1] - sensing_distance,
                             p[1] + sensing_distance]]
            near_prey = sensing_area[0][0] <= prey[0] <= sensing_area[0][1] and \
                         sensing_area[1][0] <= prey[1] <= sensing_area[1][1]
            if near_prey:
                prey_timestamps.append(i)
                break
    # Check prey near at each step and add to timestamps.
    null_timestamps = predator_timestamps + wall_timestamps + prey_timestamps
    null_timestamps = set(null_timestamps)
    desired_timestamps = [i for i in range(len(data["behavioural choice"])) if i not in null_timestamps]
    action_sequences = []
    current_action_sequence = []
    previous_point = 0
    for ts in desired_timestamps:
        if ts - 1 == previous_point:
            current_action_sequence.append(data["behavioural choice"][ts])
            previous_point = ts
        else:
            if previous_point == 0:
                current_action_sequence.append(data["behavioural choice"][ts])
                previous_point = ts
            else:
                action_sequences.append(current_action_sequence)
                current_action_sequence = [data["behavioural choice"][ts]]
                previous_point = ts
    if len(current_action_sequence) > 0:
        action_sequences.append(current_action_sequence)
    return action_sequences


def extract_exploration_action_sequences_with_positions(data, possible_visual_range=200, allowed_proximity_to_wall=200, environment_size=1500):
    """Returns all action sequences that occur n steps before consumption, with behavioural choice and """

    fish_prey_vectors = np.array([data["fish_position"]-data["prey_positions"][:, i, :] for i in range(data["prey_positions"].shape[1])])
    prey_distances = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    within_visual_range = (prey_distances < possible_visual_range) * 1
    steps = within_visual_range.shape[1]
    within_range_of_wall = ((data["fish_position"] < allowed_proximity_to_wall) * 1) + ((data["fish_position"] > environment_size - allowed_proximity_to_wall) * 1)
    within_range_of_wall = within_range_of_wall[:, 0] + within_range_of_wall[:, 1]
    within_range_of_wall_timestamps = [i for i, v in enumerate(list(within_range_of_wall)) if v > 0]

    predator_timestamps = [i for i, a in enumerate(data["predator_presence"]) if a == 1]

    exploration_timestamps = [i for i in range(steps) if np.sum(within_visual_range[:, i]) == 0]
    exploration_timestamps = [t for t in exploration_timestamps if t not in predator_timestamps]
    exploration_timestamps = [t for t in exploration_timestamps if t not in within_range_of_wall_timestamps]
    exploration_timestamps_compiled = []
    fish_positions_compiled = []
    actions_compiled = []

    current_sequence = []
    actions = []
    fish_positions = []
    for i, t in enumerate(exploration_timestamps):
        if i == 0:
            current_sequence.append(t)
            fish_positions.append(data["fish_position"][t])
            actions.append(data["action"][t])
        else:
            if t-1 != exploration_timestamps[i-1] and len(current_sequence) > 0:
                exploration_timestamps_compiled.append(np.array(current_sequence))
                actions_compiled.append(np.array(actions))
                fish_positions_compiled.append(np.array(fish_positions))

                actions = []
                fish_positions = []
                current_sequence = []

                fish_positions.append(data["fish_position"][t])
                actions.append(data["action"][t])
                current_sequence.append(t)
            else:
                current_sequence.append(t)
                fish_positions.append(data["fish_position"][t])
                actions.append(data["action"][t])

    exploration_timestamps_compiled.append(np.array(current_sequence))
    actions_compiled.append(np.array(actions))
    fish_positions_compiled.append(np.array(fish_positions))

    return exploration_timestamps_compiled, actions_compiled, fish_positions_compiled


def get_exploration_timestamps(model_name, assay_config, assay_id, n, data_cutoff=None, flatten=False):
    all_exploration_sequences = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        if data_cutoff is not None:
            data = {key: data[key][:data_cutoff] for key in data.keys()}
        if flatten:
            all_exploration_sequences = all_exploration_sequences + \
                                        extract_exploration_action_sequences_with_positions(data)[0]
        else:
            all_exploration_sequences.append(extract_exploration_action_sequences_with_positions(data)[0])
    return all_exploration_sequences


def get_exploration_sequences(model_name, assay_config, assay_id, n, data_cutoff=None, flatten=True):
    all_exploration_sequences = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        if data_cutoff is not None:
            data = {key: data[key][:data_cutoff] for key in data.keys()}
        sequences = extract_exploration_action_sequences_with_positions(data)[1]
        if flatten:
            all_exploration_sequences = all_exploration_sequences + sequences
        else:
            all_exploration_sequences.append(sequences)
    return all_exploration_sequences


def extract_exploration_action_sequences_with_fish_angles(data, possible_visual_range=100, allowed_proximity_to_wall=200, environment_size=1500):
    """Returns all action sequences that occur n steps before consumption, with behavioural choice and """

    fish_prey_vectors = np.array([data["fish_position"]-data["prey_positions"][:, i, :] for i in range(data["prey_positions"].shape[1])])
    prey_distances = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    within_visual_range = (prey_distances < possible_visual_range) * 1
    steps = within_visual_range.shape[1]
    within_range_of_wall = ((data["fish_position"] < allowed_proximity_to_wall) * 1) + ((data["fish_position"] > environment_size - allowed_proximity_to_wall) * 1)
    within_range_of_wall = within_range_of_wall[:, 0] + within_range_of_wall[:, 1]
    within_range_of_wall_timestamps = [i for i, v in enumerate(list(within_range_of_wall)) if v > 0]

    predator_timestamps = [i for i, a in enumerate(data["predator_presence"]) if a == 1]

    exploration_timestamps = [i for i in range(steps) if np.sum(within_visual_range[:, i]) == 0]
    exploration_timestamps = [t for t in exploration_timestamps if t not in predator_timestamps]
    exploration_timestamps = [t for t in exploration_timestamps if t not in within_range_of_wall_timestamps]
    exploration_timestamps_compiled = []
    fish_orientations_compiled = []
    actions_compiled = []

    current_sequence = []
    actions = []
    fish_orientations = []
    for i, t in enumerate(exploration_timestamps):
        if i == 0:
            current_sequence.append(t)
            fish_orientations.append(data["fish_angle"][t])
            actions.append(data["action"][t])
        else:
            if t-1 != exploration_timestamps[i-1] and len(current_sequence) > 0:
                exploration_timestamps_compiled.append(np.array(current_sequence))
                actions_compiled.append(np.array(actions))
                fish_orientations_compiled.append(np.array(fish_orientations))

                actions = []
                fish_positions = []
                current_sequence = []
                fish_orientations = []

                fish_orientations.append(data["fish_angle"][t])
                actions.append(data["action"][t])
                current_sequence.append(t)
            else:
                current_sequence.append(t)
                fish_orientations.append(data["fish_angle"][t])
                actions.append(data["action"][t])

    exploration_timestamps_compiled.append(np.array(current_sequence))
    actions_compiled.append(np.array(actions))
    fish_orientations_compiled.append(np.array(fish_orientations))

    return exploration_timestamps_compiled, actions_compiled, fish_orientations_compiled


def extract_no_prey_stimuli_sequences(data, assumed_prey_stim_level=23, assumed_red_stim_level=10):
    steps = data["observation"].shape[0]
    timestamps = []
    # all_uv_stim = np.array([])
    for s in range(steps):
        uv_channel = data["observation"][s, :, 1, :]
        red_channel = data["observation"][s, :, 0, :]
        # all_uv_stim = np.concatenate((all_uv_stim, np.reshape(red_channel, (-1))))
        prey_stimuli_present = (uv_channel > assumed_prey_stim_level) * 1
        wall_stimuli_present = (red_channel > assumed_red_stim_level) * 1
        if np.sum(prey_stimuli_present) + np.sum(wall_stimuli_present) > 0:
            pass
        else:
            timestamps.append(s)

    # Validation
    # import matplotlib.pyplot as plt
    # plt.hist(all_uv_stim, bins=100)
    # plt.show()
    # positions = [x for i, x in enumerate(data["fish_position"]) if i in timestamps]
    # plt.scatter([p[0] for p in positions], [p[1] for p in positions])
    # plt.show()

    all_action_sequences = []
    current_sequence = []
    # positions = [x for i, x in enumerate(data["fish_position"]) if i in timestamps]
    for i, t in enumerate(timestamps):
        if i == 0:
            current_sequence.append(data["action"][t])
        else:
            if t-1 == timestamps[i-1] or t-2 == timestamps[i-1]:
                current_sequence.append(data["action"][t])
            else:
                all_action_sequences.append(current_sequence)
                current_sequence = []
                current_sequence.append(data["action"][t])
    all_action_sequences.append(current_sequence)
    return all_action_sequences


def return_no_prey_stimuli_sequences(data, assumed_prey_stim_level=23, assumed_red_stim_level=10):
    steps = data["observation"].shape[0]
    timestamps = []
    # all_uv_stim = np.array([])
    for s in range(steps):
        uv_channel = data["observation"][s, :, 1, :]
        red_channel = data["observation"][s, :, 0, :]
        # all_uv_stim = np.concatenate((all_uv_stim, np.reshape(red_channel, (-1))))
        prey_stimuli_present = (uv_channel > assumed_prey_stim_level) * 1
        wall_stimuli_present = (red_channel > assumed_red_stim_level) * 1
        if np.sum(prey_stimuli_present) + np.sum(wall_stimuli_present) > 0:
            pass
        else:
            timestamps.append(s)

    # Validation
    # import matplotlib.pyplot as plt
    # plt.hist(all_uv_stim, bins=100)
    # plt.show()
    # positions = [x for i, x in enumerate(data["fish_position"]) if i in timestamps]
    # plt.scatter([p[0] for p in positions], [p[1] for p in positions])
    # plt.show()

    all_action_sequences = []
    current_sequence = []
    # positions = [x for i, x in enumerate(data["fish_position"]) if i in timestamps]
    for i, t in enumerate(timestamps):
        if i == 0:
            current_sequence.append(data["action"][t])
        else:
            if t-1 == timestamps[i-1] or t-2 == timestamps[i-1]:
                current_sequence.append(data["action"][t])
            else:
                all_action_sequences.append(current_sequence)
                current_sequence = []
                current_sequence.append(data["action"][t])
    all_action_sequences.append(current_sequence)
    return all_action_sequences


def get_no_prey_stimuli_sequences(model_name, assay_config, assay_id, n, data_cutoff=None):
    """Gathers all sequences (timestamps, actions) where UV channel shows no prey stimuli."""
    all_exploration_sequences = []
    for i in range(1, n + 1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        if data_cutoff is not None:
            data = {key: data[key][:data_cutoff] for key in data.keys()}
        all_exploration_sequences = all_exploration_sequences + extract_no_prey_stimuli_sequences(data)
    return all_exploration_sequences, None


def get_exploration_sequences_with_energy_state(model_name, assay_config, assay_id, n):
    all_exploration_sequences = []
    all_energy_states_by_sequence = []
    for i in range(1, n + 1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        new_energy_states = []
        new_timestamps, new_sequences, _ = extract_exploration_action_sequences_with_positions(data)
        for seq in new_timestamps:
            new_energy_states += [[data["energy_state"][i] for i in seq]]
        all_exploration_sequences = all_exploration_sequences + new_sequences
        all_energy_states_by_sequence += new_energy_states
    return all_exploration_sequences, all_energy_states_by_sequence


def label_exploration_sequences_no_prey(data, assumed_prey_stim_level=23, assumed_red_stim_level=10):
    steps = data["observation"].shape[0]
    timestamps = []
    # all_uv_stim = np.array([])
    for s in range(steps):
        uv_channel = data["observation"][s, :, 1, :]
        red_channel = data["observation"][s, :, 0, :]
        # all_uv_stim = np.concatenate((all_uv_stim, np.reshape(red_channel, (-1))))
        prey_stimuli_present = (uv_channel > assumed_prey_stim_level) * 1
        wall_stimuli_present = (red_channel > assumed_red_stim_level) * 1
        if np.sum(prey_stimuli_present) + np.sum(wall_stimuli_present) > 0:
            pass
        else:
            timestamps.append(s)
    is_exploration = [1 if i in timestamps else 0 for i in range(len(data["consumed"]))]
    return np.array(is_exploration)


def label_exploration_sequences_free_swimming(data, possible_visual_range=100, allowed_proximity_to_wall=200,
                                              environment_size=1500):
    fish_prey_vectors = np.array([data["fish_position"]-data["prey_positions"][:, i, :] for i in range(data["prey_positions"].shape[1])])
    prey_distances = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    within_visual_range = (prey_distances < possible_visual_range) * 1
    steps = within_visual_range.shape[1]
    within_range_of_wall = ((data["fish_position"] < allowed_proximity_to_wall) * 1) + ((data["fish_position"] > environment_size - allowed_proximity_to_wall) * 1)
    within_range_of_wall = within_range_of_wall[:, 0] + within_range_of_wall[:, 1]
    within_range_of_wall_timestamps = [i for i, v in enumerate(list(within_range_of_wall)) if v > 0]

    predator_timestamps = [i for i, a in enumerate(data["predator_presence"]) if a == 1]

    exploration_timestamps = [i for i in range(steps) if np.sum(within_visual_range[:, i]) == 0]
    exploration_timestamps = [t for t in exploration_timestamps if t not in predator_timestamps]
    exploration_timestamps = [t for t in exploration_timestamps if t not in within_range_of_wall_timestamps]

    is_exploration = [1 if i in exploration_timestamps else 0 for i in range(len(data["consumed"]))]
    return np.array(is_exploration)


def label_exploration_sequences_no_prey_multiple_trials(model_name, assay_config, assay_id, n):
    compiled_no_prey_exploration = []
    for trial in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{trial}")
        no_prey_exploration = label_exploration_sequences_no_prey(data)
        compiled_no_prey_exploration.append(no_prey_exploration)
    return compiled_no_prey_exploration


def label_exploration_sequences_free_swimming_multiple_trials(model_name, assay_config, assay_id, n):
    compiled_free_swimming_exploration = []
    for trial in range(1, n + 1):
        data = load_data(model_name, assay_config, f"{assay_id}-{trial}")
        free_swimming_exploration = label_exploration_sequences_free_swimming(data)
        compiled_free_swimming_exploration.append(free_swimming_exploration)
    return compiled_free_swimming_exploration
