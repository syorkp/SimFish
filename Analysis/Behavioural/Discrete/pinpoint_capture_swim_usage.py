import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.show_action_sequence_block import display_all_sequences


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
    reduced_sequences = []
    reduced_fish_positions = []
    for i, sequence in enumerate(actions_compiled):
        fish_prey_vectors = prey_positions_compiled[i][-n2:] - np.expand_dims(fish_positions_compiled[i][-n2:], 1)
        fish_prey_distances = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
        prey_within_range = (fish_prey_distances < visual_range)
        if np.sum(prey_within_range) > 0:
            pass
        else:
            reduced_sequences.append(sequence)
            reduced_fish_positions.append(fish_positions_compiled[i])
    return reduced_sequences, reduced_fish_positions


def get_capture_sequences_without_multiple_prey(model_name, assay_config, assay_id, n):

    all_reduced_sequences = []
    all_reduced_fish_positions = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        prey_capture_timestamps_compiled, consumption_timestamps, actions_compiled, fish_positions_compiled, prey_positions_compiled = extract_capture_sequences_with_additional_steps(data)
        reduced_sequences, reduced_fish_positions = remove_sequences_with_nearby_prey_following_capture(actions_compiled, fish_positions_compiled, prey_positions_compiled)
        all_reduced_sequences = all_reduced_sequences + reduced_sequences
        all_reduced_fish_positions = all_reduced_fish_positions + reduced_fish_positions

    return all_reduced_sequences, all_reduced_fish_positions


def extract_max_uv_stimuli_with_scs(data):
    actions = data["action"]
    capture_timestamps = [t for t, a in enumerate(data["action"]) if a == 3]
    max_uv_stimuli = [np.max(data["observation"][t]) for t in capture_timestamps]
    return max_uv_stimuli


def get_max_uv_stimuli_with_scs(model_name, assay_config, assay_id, n):
    max_uv_stimuli = []

    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        max_uv_stimuli = max_uv_stimuli + extract_max_uv_stimuli_with_scs(data)

    return max_uv_stimuli


for i in range(1, 5):
    # Extracting based on prey positions in period following capture:
    reduced_sequences, reduced_fish_positions = get_capture_sequences_without_multiple_prey(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10)

    # Display as block:
    display_all_sequences(reduced_sequences)

    # Max UV stimulus required to elicit capture swims
    max_stims = get_max_uv_stimuli_with_scs(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10)
    plt.hist(max_stims, 100)
    plt.title("Frequency Histogram sCS usage with UV Max Stimulation")
    plt.ylabel("Frequency")
    plt.xlabel("Max UV Photons")
    plt.xlim(0, 255)
    plt.show()


