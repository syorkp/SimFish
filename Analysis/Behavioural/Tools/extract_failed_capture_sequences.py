import numpy as np

from Analysis.load_data import load_data


def extract_failed_capture_sequences(data, n=20, strict=True):
    """Returns all action sequences that occur n steps before consumption"""
    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    capture_swim_timestamps = [i for i, a in enumerate(data["action"]) if a == 3]
    if strict:
        successful_buffer = 10
        safe_timestamps = []
        low_bounds = np.array(consumption_timestamps) - successful_buffer/2
        high_bounds = np.array(consumption_timestamps) - successful_buffer/2
        for t in range(len(data["action"])):
            if t
    else:
        failed_capture_timestamps = [i for i in capture_swim_timestamps if i not in consumption_timestamps]

    prey_c_t = []
    action_sequences = []
    while len(consumption_timestamps) > 0:
        index = consumption_timestamps.pop(0)
        prey_capture_timestamps = [i for i in range(index-n+1, index+1) if i >= 0]
        prey_c_t.append(prey_capture_timestamps)
        action_sequence = [data["action"][i] for i in prey_capture_timestamps]
        action_sequences.append(action_sequence)
    return action_sequences, prey_c_t


def get_failed_capture_sequences(model_name, assay_config, assay_id, n):
    all_failed_capture_sequences = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_capture_sequences = all_failed_capture_sequences + extract_failed_capture_sequences(data)[0]
    return all_failed_capture_sequences
