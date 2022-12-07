

import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.show_observation_sequences import display_obs_sequence


def get_steps_in_motion(data, motion_threshold=0.1, rotation=True):
    x_diff = data["fish_position"][1:, 0] - data["fish_position"][:-1, 0]
    y_diff = data["fish_position"][1:, 1] - data["fish_position"][:-1, 1]

    distance_moved = (x_diff ** 2 + y_diff ** 2) ** 0.5
    superthreshold = distance_moved >= motion_threshold
    if not rotation:
        chosen_fish_angle = data["fish_angle"][1:] - data["fish_angle"][:-1]
        no_rotation = np.absolute(chosen_fish_angle) < 0.15
        superthreshold *= no_rotation
    # print(np.sum(superthreshold * 1))
    steps = [i for i, v in enumerate(superthreshold) if v]

    sequence_timestamps = []
    current_seq = []
    for i, t in enumerate(steps):
        if i == 0:
            current_seq.append(t)
        else:
            if t - 1 == steps[i - 1]:
                current_seq.append(t)
            else:
                sequence_timestamps.append(current_seq)
                current_seq = []
    if len(current_seq) > 0:
        sequence_timestamps.append(current_seq)

    return sequence_timestamps


if __name__ == "__main__":
    observations = []
    for i in range(1, 11):
        d = load_data("dqn_scaffold_14-2", "Behavioural-Data-Free", f"Naturalistic-{i}")
        steps = get_steps_in_motion(d, rotation=False)
        long_enough_steps = [s for s in steps if len(s) > 10]
        for seq in long_enough_steps:
            observations += [d["observation"][seq, :, :, :]]
    for observation in observations:
        display_obs_sequence(observation)
