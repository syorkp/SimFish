import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.show_observation_sequences import display_obs_sequence


def get_all_pred_observation_sequences(model_name, assay_config, assay_id, n, terminal):
    obs_sequences_compiled = []

    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
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

        if terminal and len(pred_sequence_timestamps) > 0:
            obs_sequences_compiled.append([d["observation"][s] for s in pred_sequence_timestamps[-1]])
        else:
            for seq in pred_sequence_timestamps:
                obs_sequences_compiled.append([d["observation"][s] for s in seq])

    return obs_sequences_compiled


if __name__ == "__main__":
    obs_seq = get_all_pred_observation_sequences("dqn_scaffold_26-1", "Behavioural-Data-Videos-C1", "Naturalistic", 5,
                                                 terminal=True)
    for seq in obs_seq:
        display_obs_sequence(seq)
