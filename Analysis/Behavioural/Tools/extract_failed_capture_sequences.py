import numpy as np

from Analysis.load_data import load_data


def extract_failed_capture_sequences(data, n=20, strict=True):
    """Returns all action sequences that occur n steps before consumption"""
    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    capture_swim_timestamps = [i for i, a in enumerate(data["action"]) if a == 3]
    if strict:
        failed_capture_timestamps = []
        successful_buffer = 10
        for t in consumption_timestamps:
            for s in capture_swim_timestamps:
                if t - successful_buffer/2 > s or t + successful_buffer/2 < s:
                    continue
                else:
                    failed_capture_timestamps.append(s)
        failed_capture_timestamps = list(set(failed_capture_timestamps))
    else:
        failed_capture_timestamps = [i for i in capture_swim_timestamps if i not in consumption_timestamps]
    failed_c_t = []
    action_sequences = []
    while len(failed_capture_timestamps) > 0:
        index = failed_capture_timestamps.pop(0)
        failed_capture_timestamps = [i for i in range(index-n+1, index+1) if i >= 0]
        failed_c_t.append(failed_capture_timestamps)
        action_sequence = [data["action"][i] for i in failed_capture_timestamps]
        action_sequences.append(action_sequence)
    return action_sequences, failed_c_t


def get_failed_capture_sequences(model_name, assay_config, assay_id, n):
    all_failed_capture_sequences = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_failed_capture_sequences = all_failed_capture_sequences + extract_failed_capture_sequences(data)[0]
    return all_failed_capture_sequences
