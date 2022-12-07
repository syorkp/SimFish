import numpy as np

from Analysis.load_data import load_data

from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import extract_exploration_action_sequences_with_fish_angles


"""
Script able to return time points where a directional switch occurs
"""


def label_turn_switch_timepoints(action_sequences, associated_timesteps, threshold_for_chain=2):
    left_turns  = np.array([1, 4]).tolist()
    right_turns = np.array([2, 5]).tolist()


    all_switch_timestamps = []

    for seq, ts in zip(action_sequences, associated_timesteps):
        directional_streak = 0
        current_direction = None
        new_direction = None

        for i, a in enumerate(seq):
            if a not in left_turns and a not in right_turns:
                pass
            else:
                if a in left_turns:
                    new_direction = "L"
                elif a in right_turns:
                    new_direction = "R"

                directional_streak += 1

                if current_direction is not None:
                    if current_direction != new_direction:
                        if directional_streak > threshold_for_chain:
                            all_switch_timestamps.append(ts[i])
                        directional_streak = 0
                current_direction = new_direction
    return all_switch_timestamps


if __name__ == "__main__":
    data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic-1")
    timestamps, sequences, _ = extract_exploration_action_sequences_with_fish_angles(data)
    turn_switch_timestamps = label_turn_switch_timepoints(sequences, timestamps)

