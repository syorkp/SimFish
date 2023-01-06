import numpy as np

from Analysis.load_data import load_data


"""
Scripts to extract hunting sequences as defined by Henriques et al. (2019):
   - paramecium located within reactive perceptive field <6mm and within 120 degrees
   - distance-gain and orientation-gain for the first two bouts of the hunting sequence were positive with respect to 
   that prey item.
"""


def get_hunting_sequences_timestamps(data, successful_captures, sand_grain_version=False):
    """

    :param data:
    :param successful_captures: Boolean to discard identified sequences if they dont end in a successful capture.
    :param sand_grain_version: Boolean to use on sand grain positions, rather than prey.
    :return:
    """

    if sand_grain_version:
        prey_positions = data["sand_grain_positions"]
    else:
        prey_positions = data["prey_positions"]
    fish_positions = data["fish_position"]
    fish_orientation = data["fish_angle"]
    fish_orientation_sign = ((fish_orientation >= 0) * 1) + ((fish_orientation < 0) * -1)
    fish_orientation %= 2 * np.pi * fish_orientation_sign

    fish_prey_vectors = prey_positions - np.expand_dims(fish_positions, 1)
    fish_prey_distance = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    within_six_mm = (fish_prey_distance < 60)

    # fish_prey_angles = np.arctan(fish_prey_vectors[:, :, 1] / fish_prey_vectors[:, :, 0])# - np.expand_dims(fish_orientation, 1)

    # Adjust according to quadrents.
    fish_prey_angles = np.arctan(fish_prey_vectors[:, :, 1] / fish_prey_vectors[:, :, 0])

    #   Generates positive angle from left x axis clockwise.
    unchanged = np.ones(fish_prey_angles.shape).astype(int)
    # UL quadrent
    in_ul_quadrent = (fish_prey_vectors[:, :, 0] < 0) * (fish_prey_vectors[:, :, 1] > 0)
    fish_prey_angles[in_ul_quadrent] += np.pi
    # BR quadrent
    in_br_quadrent = (fish_prey_vectors[:, :, 0] > 0) * (fish_prey_vectors[:, :, 1] < 0)
    fish_prey_angles[in_br_quadrent] += (np.pi * 2)
    # BL quadrent
    in_bl_quadrent = (fish_prey_vectors[:, :, 0] < 0) * (fish_prey_vectors[:, :, 1] < 0)
    fish_prey_angles[in_bl_quadrent] += np.pi
    # Angle ends up being between 0 and 2pi as clockwise from right x axis. Same frame as fish angle:

    fish_prey_incidence = np.absolute(np.expand_dims(fish_orientation, 1) - fish_prey_angles)
    fish_prey_incidence[fish_prey_incidence > np.pi] -= 2 * np.pi
    fish_prey_incidence = np.absolute(fish_prey_incidence)

    within_visual_field = (fish_prey_incidence < (np.pi * (120/180))) + (np.pi * 2 - fish_prey_incidence < (np.pi * (120/180)))
    within_visual_field[within_visual_field > 0] = 1  # To normalise

    paramecium_in_zone = within_six_mm * within_visual_field

    # Convert to timestamps (with prey indices)
    num_paramecia = paramecium_in_zone.shape[1]
    paramecia_timestamps_sequences = []
    for p in range(num_paramecia):
        timestamps = np.argwhere(paramecium_in_zone[:, p])[:, 0]

        sequences = []
        current_sequence = []
        # Convert timestamps to sequences (only keeping those that meet the second condition_
        for i, t in enumerate(timestamps):
            if i == 0:
                current_sequence.append(t)
            else:
                if t - 1 == timestamps[i-1]:
                    current_sequence.append(t)
                else:
                    if len(current_sequence) >= 2:
                        # first_ts = current_sequence[:2]
                        # first_ts.insert(0, first_ts[0]-1)

                        # If distance gain is positive
                        distance_to_prey_sequence = fish_prey_distance[current_sequence, p]
                        distance_gain = distance_to_prey_sequence[1:] - distance_to_prey_sequence[:-1]

                        # if angle gain is positive
                        fish_prey_angles_sequence = fish_prey_incidence[current_sequence, p]
                        angle_gain = fish_prey_angles_sequence[1:] - fish_prey_angles_sequence[:-1]

                        permitted_gain = (distance_gain < 0) * (angle_gain < 0)

                        # Find earliest two steps where distance and angle gains are decreasing. This forms the start of the sequence
                        for s in range(permitted_gain.shape[0]-1):
                            if np.all(permitted_gain[s:s+1]):
                                sequences.append(current_sequence[s:])
                                break
                    current_sequence = [t]
        if len(current_sequence) > 0:
            sequences.append(current_sequence)
        paramecia_timestamps_sequences.append(sequences)

    all_timestamps = [seq for sequen in paramecia_timestamps_sequences for seq in sequen]
    all_actions = []
    all_used_timestamps = []

    for seq in all_timestamps:
        if successful_captures:
            endings = [s + 1 for s in seq]
            consumptions = data["consumed"][endings]
            index = 0  # Variable to keep track of sequences if multiple ones are found within sequence
            for i, c in enumerate(consumptions):
                if c:
                    all_actions.append(data["action"][seq[index:i+2]])
                    all_used_timestamps.append(seq[index:i+2])
                    index = i + 2
        else:
            all_actions.append(data["action"][seq])
            all_used_timestamps.append(seq)

    return all_actions, all_used_timestamps


if __name__ == "__main__":
    d = load_data("dqn_beta-1", "Behavioural-Data-Free", "Naturalistic-2")
    seq, ts = get_hunting_sequences_timestamps(d, True)
    all_seq, all_ts = get_hunting_sequences_timestamps(d, False)
    hunt_endpoints = [i for i, c in enumerate(d["consumed"]) if c]
    # Validate by labelling successful captures.
