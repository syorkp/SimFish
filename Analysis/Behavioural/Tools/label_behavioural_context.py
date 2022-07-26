import numpy as np

from Analysis.Behavioural.Tools.extract_capture_sequences import label_capture_sequences
from Analysis.Behavioural.Tools.extract_escape_sequences import label_escape_sequences
from Analysis.Behavioural.Tools.extract_exploration_sequences import label_exploration_sequences_no_prey, \
    label_exploration_sequences_free_swimming
from Analysis.load_data import load_data


def label_behavioural_context(data, environment_size):
    """Return list of lists of labels for behavioural context at each step.
        - 0 No recognised context
        - 1 Prey capture behaviour
        - 2 Exploration behaviour
        - 3 Escape behaviour
        - 4 Exploration behaviour 2
        - 5 Wall interaction
    """
    capture_ts = label_capture_sequences(data, n=20) * 1
    predator_avoidance_ts = label_escape_sequences(data) * 3
    exploration_np_ts = label_exploration_sequences_no_prey(data) * 4
    exploration_fs_ts = label_exploration_sequences_free_swimming(data, environment_size=environment_size) * 2

    capture_ts = np.expand_dims(capture_ts, 1)
    predator_avoidance_ts = np.expand_dims(predator_avoidance_ts, 1)
    exploration_np_ts = np.expand_dims(exploration_np_ts, 1)
    exploration_fs_ts = np.expand_dims(exploration_fs_ts, 1)

    behavioural_context_label = np.concatenate((capture_ts, predator_avoidance_ts, exploration_np_ts, exploration_fs_ts), axis=1)
    return behavioural_context_label


def label_behavioural_context_multiple_trials(datas, environment_size):
    associated_behavioural_context_labels = []

    for data in datas:
        associated_behavioural_context_labels.append(label_behavioural_context(data, environment_size))

    return associated_behavioural_context_labels


if __name__ == "__main__":
    data = load_data("dqn_scaffold_18-2", "Behavioural-Data-Free", f"Naturalistic-2")

    label_behavioural_context(data, 1500)





