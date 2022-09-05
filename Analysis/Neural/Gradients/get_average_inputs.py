from collections import Counter
import numpy as np

from Analysis.load_data import load_data


def get_most_common_network_inputs(observations, rnn_state, energy_state, salt, actions, in_light):
    mean_observation = np.mean(observations, axis=(0))
    mean_rnn_state = np.mean(rnn_state, axis=(0))
    mean_rnn_state = (mean_rnn_state[0], mean_rnn_state[1])
    mean_energy_state = np.mean(energy_state)
    mean_salt_input = np.mean(salt)

    action_bin_counts = Counter(actions)
    inputted_action = action_bin_counts.most_common(1)[0][0]

    in_light_bin_counts = Counter(in_light)
    inputted_in_light = in_light_bin_counts.most_common(1)[0][0]

    return mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light


def get_most_common_network_inputs_from_data(data):
    # TODO: make this work for multiple datas
    mean_observation = np.mean(data["observation"], axis=(0))
    mean_rnn_state = np.mean(data["rnn_state_actor"], axis=(0))
    mean_rnn_state = (mean_rnn_state[0], mean_rnn_state[1])
    mean_energy_state = np.mean(data["energy_state"])
    mean_salt_input = np.mean(data["salt"])

    action_bin_counts = Counter(data["action"])
    inputted_action = action_bin_counts.most_common(1)[0][0]

    in_light_bin_counts = Counter(data["in_light"])
    inputted_in_light = in_light_bin_counts.most_common(1)[0][0]

    return mean_observation, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light, mean_rnn_state


def get_average_input_during_context(data, labels):
    """Takes one hot encoded event and returns the average input during this period."""
    labels = np.array(labels)

    observations = data["observation"][labels == 1]
    rnn_state = data["rnn_state_actor"][labels == 1]
    salt = data["salt"][labels == 1]
    energy_state = data["energy_state"][labels == 1]
    actions = data["action"][labels == 1]
    in_light = data["in_light"][labels == 1]

    mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light = \
        get_most_common_network_inputs(observations, rnn_state, energy_state, salt, actions, in_light)

    return mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light



def get_average_input_during_context_multiple_trials(datas, labels):
    """Takes one hot encoded event and returns the average input during this period."""
    labels = np.array(labels)

    observations
    rnn_state
    salt
    energy_state
    actions
    in_light

    for data, label in zip(datas, labels):
        observations = data["observation"][labels == 1]
        rnn_state = data["rnn_state_actor"][labels == 1]
        salt = data["salt"][labels == 1]
        energy_state = data["energy_state"][labels == 1]
        actions = data["action"][labels == 1]
        in_light = data["in_light"][labels == 1]

    mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light = \
        get_most_common_network_inputs(observations, rnn_state, energy_state, salt, actions, in_light)

    return mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light


if __name__ == "__main__":
    data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic-1")
    x = get_average_input_during_context(data, [1 for i in range(1000)] + [0 for i in range(1000)])


