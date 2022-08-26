import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from Analysis.load_data import load_data

tf.logging.set_verbosity(tf.logging.ERROR)


def get_all_observations(model_name, assay_config, assay_id, n):
    observations = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        if i == 1:
            observations = data["observation"]
        else:
            observations = np.concatenate((observations, data["observation"]), axis=0)

    return np.array(observations)

def get_all_cnn_activity(model_name, assay_config, assay_id, n):
    keys_to_collect = ["conv1l", "conv2l", "conv3l", "conv4l", "conv1r", "conv2r", "conv3r", "conv4r"]
    collected_data = {}
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        if i == 1:
            collected_data = {key: data[key] for key in keys_to_collect}
        else:
            for key in keys_to_collect:
                collected_data[key] = np.concatenate((collected_data[key], data[key]), axis=0)
    return collected_data


if __name__ == "__main__":
    cnn_activity = get_all_cnn_activity("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 1)
    observations = get_all_observations("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 1)


