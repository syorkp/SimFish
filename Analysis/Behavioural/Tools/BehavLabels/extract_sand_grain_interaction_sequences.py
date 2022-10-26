import numpy as np

from Analysis.load_data import load_data


def get_sand_grain_engagement_sequences_multiple_trials(model_name, assay_config, assay_id, n, range_for_engagement,
                                                        preceding_steps):
    """Returns the numbers of possible and actual engagements with each feature - multiple trials"""

    actions_compiled = []
    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        action_sequences = get_sand_grain_engagement_sequences(d, range_for_engagement, preceding_steps)
        actions_compiled += action_sequences
    return actions_compiled


def get_sand_grain_engagement_sequences(data, range_for_engagement=50, preceding_steps=20):
    """Returns the numbers of possible and actual engagements with each feature"""

    fish_position = np.expand_dims(data["fish_position"], 1)
    sand_grain_positions = data["sand_grain_positions"]

    fsg_vectors = sand_grain_positions - fish_position

    fsg_distances = (fsg_vectors[:, :, 0] ** 2 + fsg_vectors[:, :, 1] ** 2) ** 0.5

    fsg_within_range = (fsg_distances < range_for_engagement) * 1

    interaction_timestamps = [i for i, v in enumerate(fsg_within_range) if np.any(v == 1)]
    # TODO: Pad out interaction sequences to include 10 prior steps - OR reduce so that contiguous steps dont count.

    interaction_timestamps_sequences = []
    actions_compiled = []

    current_sequence = []
    actions = []
    for i, t in enumerate(interaction_timestamps):
        if i == 0:
            current_sequence.append(t)
            actions.append(data["action"][t])
        else:
            if t-1 != interaction_timestamps[i-1] and len(current_sequence) > 0:
                actions = [data["action"][i] for i in range(current_sequence[0]-preceding_steps, current_sequence[0])] + actions
                current_sequence = [i for i in range(current_sequence[0]-preceding_steps, current_sequence[0])] + current_sequence

                interaction_timestamps_sequences.append(np.array(current_sequence))
                actions_compiled.append(actions)

                actions = []
                current_sequence = []

                actions.append(data["action"][t])
                current_sequence.append(t)
            else:
                current_sequence.append(t)
                actions.append(data["action"][t])
    if len(actions) > 0:
        actions = [data["action"][i] for i in
                   range(current_sequence[0] - preceding_steps, current_sequence[0])] + actions
        actions_compiled.append(actions)

    return actions_compiled


if __name__ == "__main__":
    d = load_data("dqn_scaffold_33-1", "Behavioural-Data-Free", "Naturalistic-1")
    get_sand_grain_engagement_sequences(d)


