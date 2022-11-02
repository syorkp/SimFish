import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.show_observation_sequences import display_obs_sequence


def get_all_pred_observation_sequences(model_name, assay_config, assay_id, n, terminal, inc_prev_steps=False):
    obs_sequences_compiled = []
    fish_predator_angles_compiled = []

    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        fish_position = d["fish_position"]
        fish_orientation = d["fish_angle"]
        predator_position = d["predator_positions"]
        pred_timestamps = [i for i, p in enumerate(d["predator_presence"]) if p == 1]
        pred_sequence_timestamps = []
        current_seq = []
        for i, t in enumerate(pred_timestamps):
            if i == 0:
                current_seq.append(t)
            else:
                if t-1 == pred_timestamps[i-1]:
                    current_seq.append(t)
                else:
                    pred_sequence_timestamps.append(current_seq)
                    current_seq = []
        if len(current_seq) > 0:
            pred_sequence_timestamps.append(current_seq)

        if inc_prev_steps:
            for i, seq in enumerate(pred_sequence_timestamps):
                pred_sequence_timestamps[i] = [j for j in range(min(seq)-10, min(seq))] + seq


        if terminal and len(pred_sequence_timestamps) > 0:
            obs_sequences_compiled.append([d["observation"][s] for s in pred_sequence_timestamps[-1]])

            f_positions = np.array([fish_position[s] for s in pred_sequence_timestamps[-1]])
            f_angles = np.array([fish_orientation[s] for s in pred_sequence_timestamps[-1]])
            p_positions = np.array([predator_position[s] for s in pred_sequence_timestamps[-1]])
            f_angles_scaling = f_angles + ((f_angles // (np.pi * 2)) * np.pi * -2)

            f_p_vector = f_positions - p_positions
            f_p_angle = np.arctan(f_p_vector[:, 1] / f_p_vector[:, 0])
            predators_on_left = (p_positions[:, 0] < f_positions[:, 0]) * np.pi
            f_p_angle += predators_on_left

            f_p_incidence = f_angles_scaling - f_p_angle
            # f_p_incidence_scaling = f_p_incidence + ((f_p_incidence // (np.pi * 2)) * np.pi * -2)

            too_high = f_p_incidence > np.pi
            f_p_incidence[too_high] -= np.pi * 2
            too_low = f_p_incidence < -np.pi
            f_p_incidence[too_low] += np.pi * 2

            fish_predator_angles_compiled.append(f_p_incidence)
        else:
            for seq in pred_sequence_timestamps:
                obs_sequences_compiled.append([d["observation"][s] for s in seq])

                f_positions = np.array([fish_position[s] for s in seq])
                p_positions = np.array([predator_position[s] for s in seq])
                f_angles = np.array([fish_orientation[s] for s in seq])

                f_angles_scaling = f_angles + ((f_angles // (np.pi * 2)) * np.pi * -2)

                f_p_vector = f_positions - p_positions
                f_p_angle = np.arctan(f_p_vector[:, 1] / f_p_vector[:, 0])
                predators_on_left = (p_positions[:, 0] < f_positions[:, 0]) * np.pi
                f_p_angle += predators_on_left

                f_p_incidence = f_angles_scaling - f_p_angle
                # f_p_incidence_scaling = f_p_incidence + ((f_p_incidence // (np.pi * 2)) * np.pi * -2)

                too_high = f_p_incidence > np.pi
                f_p_incidence[too_high] -= np.pi * 2
                too_low = f_p_incidence < -np.pi
                f_p_incidence[too_low] += np.pi * 2

                fish_predator_angles_compiled.append(f_p_incidence)

    return obs_sequences_compiled, fish_predator_angles_compiled


if __name__ == "__main__":
    obs_seq, angles_seq = get_all_pred_observation_sequences("dqn_scaffold_26-1", "Behavioural-Data-Predator", "Naturalistic", 6,
                                                             terminal=True, inc_prev_steps=10)
    for i, seq in enumerate(obs_seq):
        display_obs_sequence(seq, angles_seq[i])
        # plt.clf()
        # #
        # plt.plot(angles_seq[i])
        # plt.show()

        x = True
