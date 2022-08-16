import numpy as np


def get_mean_activity_events(rnn_data, labels):
    """Computes the mean activity of a neuron during one-hot encoded events."""
    mean_activity_vector = np.zeros((labels.shape[1]))
    for i in range(len(mean_activity_vector)):
        mean_activity_vector[i] = np.mean(rnn_data[labels[:, i] == 1])
    return mean_activity_vector
