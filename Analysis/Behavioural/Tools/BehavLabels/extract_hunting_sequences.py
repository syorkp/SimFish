import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_fish_prey_incidence import get_fish_prey_incidence_multiple_prey

"""
Scripts to extract hunting sequences as defined by Henriques et al. (2019):
   - paramecium located within reactive perceptive field <6mm and within 120 degrees
   - distance-gain and orientation-gain for the first two bouts of the hunting sequence were positive with respect to 
   that prey item.
"""



def get_prey_being_hunted(prey_positions, fish_positions, fish_orientation):
    """Returns one hot encoded array of (time x prey_num) where 1 means that paramecium meets Henriques (2019) definition
    of being hunted."""

    # Check within 6mm
    fish_prey_vectors = prey_positions - np.expand_dims(fish_positions, 1)
    fish_prey_distance = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    within_six_mm = (fish_prey_distance < 60)

    fish_prey_incidence = get_fish_prey_incidence_multiple_prey(fish_positions, fish_orientation, prey_positions)
    fish_prey_incidence = np.absolute(fish_prey_incidence)

    within_visual_field = (fish_prey_incidence < (np.pi * (120/180))) + (np.pi * 2 - fish_prey_incidence < (np.pi * (120/180)))
    within_visual_field[within_visual_field > 0] = 1  # To normalise

    paramecium_in_zone = within_six_mm * within_visual_field

    return paramecium_in_zone


def remove_overlapping_sequences(all_used_timestamps, all_actions):
    # Remove direct repeats
    indices_to_delete = []
    for i, ts in enumerate(all_used_timestamps):
        for ts_2 in all_used_timestamps[i+1:]:
            if ts == ts_2:
                indices_to_delete.append(i)

    for i_d in reversed(indices_to_delete):
        del all_used_timestamps[i_d]
        del all_actions[i_d]

    # Remove shorter sequences with the same terminus as a longer one
    indices_to_delete = []
    for i, ts in enumerate(all_used_timestamps):
        for j, ts_2 in enumerate(all_used_timestamps):
            if i == j:
                pass
            else:
                if ts[-1] == ts_2[-1] and len(ts) <= len(ts_2):
                    indices_to_delete.append(i)

    try:
        for i_d in reversed(indices_to_delete):
            del all_used_timestamps[i_d]
            del all_actions[i_d]
    except IndexError:
        x = True

    # try:
    #     all_actions = [x for _, x in sorted(zip(all_used_timestamps, all_actions))]
    # except ValueError:
    #     x = True
    #
    # x = True
    return all_used_timestamps, all_actions


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

    # Check within 6mm
    fish_prey_vectors = prey_positions - np.expand_dims(fish_positions, 1)
    fish_prey_distance = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    within_six_mm = (fish_prey_distance < 60)

    # fish_orientation_sign = ((fish_orientation >= 0) * 1) + ((fish_orientation < 0) * -1)
    # fish_orientation %= 2 * np.pi * fish_orientation_sign

    # fish_prey_angles = np.arctan(fish_prey_vectors[:, :, 1] / fish_prey_vectors[:, :, 0])# - np.expand_dims(fish_orientation, 1)

    # # Adjust according to quadrents.
    # fish_prey_angles = np.arctan(fish_prey_vectors[:, :, 1] / fish_prey_vectors[:, :, 0])
    #
    # #   Generates positive angle from left x axis clockwise.
    # # UL quadrent
    # in_ul_quadrent = (fish_prey_vectors[:, :, 0] < 0) * (fish_prey_vectors[:, :, 1] > 0)
    # fish_prey_angles[in_ul_quadrent] += np.pi
    # # BR quadrent
    # in_br_quadrent = (fish_prey_vectors[:, :, 0] > 0) * (fish_prey_vectors[:, :, 1] < 0)
    # fish_prey_angles[in_br_quadrent] += (np.pi * 2)
    # # BL quadrent
    # in_bl_quadrent = (fish_prey_vectors[:, :, 0] < 0) * (fish_prey_vectors[:, :, 1] < 0)
    # fish_prey_angles[in_bl_quadrent] += np.pi
    # # Angle ends up being between 0 and 2pi as clockwise from right x axis. Same frame as fish angle:
    #
    # fish_prey_incidence = np.absolute(np.expand_dims(fish_orientation, 1) - fish_prey_angles)
    # fish_prey_incidence[fish_prey_incidence > np.pi] -= 2 * np.pi

    # Check within 120 degrees in visual field.
    fish_prey_incidence = get_fish_prey_incidence_multiple_prey(fish_positions, fish_orientation, prey_positions)
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
            try:
                consumptions = data["consumed"][endings]
            except IndexError:
                # Occurs in the case where a hunting sequence continues until the last step (so consumption in the next step cant be verified)
                consumptions = data["consumed"][endings[:-1]]
            index = 0  # Variable to keep track of sequences if multiple ones are found within sequence
            for i, c in enumerate(reversed(consumptions)):
                revered_i = len(consumptions) - i
                if c:
                    all_actions.append(data["action"][seq[index:revered_i]])
                    all_used_timestamps.append(seq[index:revered_i])
                    index = i + 2
                    break
        else:
            all_actions.append(data["action"][seq])
            all_used_timestamps.append(seq)

    all_used_timestamps, all_actions = remove_overlapping_sequences(all_used_timestamps, all_actions)

    return all_actions, all_used_timestamps


def get_hunting_sequences(model_name, assay_config, assay_id, n, successful_captures, include_subsequent_action):
    all_hunting_sequences = []
    for i_trial in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i_trial}")
        successful_seq_end = [i for i, c in enumerate(data["consumed"]) if c]
        if include_subsequent_action:  # In successful sequences, this will be the final consumption
            seq, timestamps = get_hunting_sequences_timestamps(data, successful_captures)
            if successful_captures:
                if len(successful_seq_end) < len(timestamps):
                    print(f"Trial {i_trial} error in successful sequence isolation")
            for i_t, ts in enumerate(timestamps):
                try:
                    seq[i_t] = np.concatenate((seq[i_t], np.array([data["action"][ts[-1]+1]])))
                except IndexError: # Occurs in the case where a hunting sequence continues until the last step (so consumption in the next step cant be verified)
                    pass
            all_hunting_sequences = all_hunting_sequences + seq
        else:
            all_hunting_sequences = all_hunting_sequences + get_hunting_sequences_timestamps(data, successful_captures)[0]
    return all_hunting_sequences


def check_hunting_identification_success_rate(model_name, assay_config, assay_id, n):
    successful_hunting_sequences = get_hunting_sequences(model_name, assay_config, assay_id, n,
                                              successful_captures=True, include_subsequent_action=True)
    all_hunting_sequences = get_hunting_sequences(model_name, assay_config, assay_id, n,
                                              successful_captures=False, include_subsequent_action=True)

    consumptions = [np.sum(load_data(model_name, assay_config, f"{assay_id}-{i}")["consumed"]) for i in range(1, n+1)]
    total_consumptions = np.sum(consumptions)

    print(f"""Total consumptions: {total_consumptions}
Total Identified: {len(successful_hunting_sequences)}
Total Identified (inc successful): {len(all_hunting_sequences)}

Success rate: {round(len(successful_hunting_sequences)/total_consumptions * 100)}%""")



if __name__ == "__main__":
    # d = load_data("dqn_beta-1", "Behavioural-Data-Free", "Naturalistic-2")
    # seq, ts = get_hunting_sequences_timestamps(d, True)
    # all_seq, all_ts = get_hunting_sequences_timestamps(d, False)
    # hunt_endpoints = [i for i, c in enumerate(d["consumed"]) if c]
    # Validate by labelling successful captures.
    check_hunting_identification_success_rate(f"dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic", 100)