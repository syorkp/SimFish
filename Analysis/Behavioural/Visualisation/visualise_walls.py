import numpy as np
from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.show_observation_sequences import display_obs_sequence


def show_wall_interaction_sequence(model_name, assay_config, assay_id, n, env_w=3000, env_h=3000, buffer=100):
    obs_sequences = []

    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        in_proximity = (d["fish_position"][:, 0] < buffer) + \
                       (d["fish_position"][:, 0] > env_w - buffer) + \
                       (d["fish_position"][:, 1] < buffer) + \
                       (d["fish_position"][:, 1] > env_h - buffer)
        steps_in_proximity = [i for i, p in enumerate(in_proximity) if p]

        sequence_timestamps = []
        current_seq = []
        for i, t in enumerate(steps_in_proximity):
            if i == 0:
                current_seq.append(t)
            else:
                if t - 1 == steps_in_proximity[i - 1]:
                    current_seq.append(t)
                else:
                    sequence_timestamps.append(current_seq)
                    current_seq = []
        if len(current_seq) > 0:
            sequence_timestamps.append(current_seq)
        # sequence_timestamps = [[i for i in range(len(d["observation"]))]]

        for seq in sequence_timestamps:
            obs_sequences.append([d["observation"][s] for s in seq])

    for seq in obs_sequences:
        display_obs_sequence(seq)


if __name__ == "__main__":
    show_wall_interaction_sequence("dqn_scaffold_beta_test-1", "Behavioural-Data-Videos-A1", "Naturalistic", 5)


