from collections import Counter
import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.label_behavioural_context import label_behavioural_context_multiple_trials


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

    return mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light


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


def get_average_input_during_context_multiple_trials(datas, labels_compiled):
    """Takes one hot encoded event and returns the average input during this period."""
    labels_compiled = np.array(labels_compiled)

    compiled_observations = np.zeros((0, datas[0]["observation"].shape[1], datas[0]["observation"].shape[2],
                                      datas[0]["observation"].shape[3]))
    compiled_rnn_state = np.zeros((0, datas[0]["rnn_state_actor"].shape[1], datas[0]["rnn_state_actor"].shape[2],
                                   datas[0]["rnn_state_actor"].shape[3]))
    compiled_salt = np.zeros((0))
    compiled_energy_state = np.zeros((0))
    compiled_actions = np.zeros((0))
    compiled_in_light = np.zeros((0))

    for data, labels in zip(datas, labels_compiled):
        observations = data["observation"][labels == 1]
        rnn_state = data["rnn_state_actor"][labels == 1]
        salt = data["salt"][labels == 1]
        energy_state = data["energy_state"][labels == 1]
        actions = data["action"][labels == 1]
        in_light = data["in_light"][labels == 1]

        compiled_observations = np.concatenate((compiled_observations, observations), axis=0)
        compiled_rnn_state = np.concatenate((compiled_rnn_state, rnn_state), axis=0)
        compiled_salt = np.concatenate((compiled_salt, salt), axis=0)
        compiled_energy_state = np.concatenate((compiled_energy_state, energy_state), axis=0)
        compiled_actions = np.concatenate((compiled_actions, actions), axis=0)
        compiled_in_light = np.concatenate((compiled_in_light, in_light), axis=0)

    mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light = \
        get_most_common_network_inputs(compiled_observations,
                                       compiled_rnn_state,
                                       compiled_energy_state,
                                       compiled_salt,
                                       compiled_actions,
                                       compiled_in_light)

    mean_observation = np.around(mean_observation).astype(int)

    return mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light


def get_mean_inputs_for_context(model_name, assay_config, assay_id, n, context):
    datas_compiled = []

    for i in range(1, n+1):
        datas_compiled.append(load_data(model_name, assay_config, f"{assay_id}-{i}"))

    labels = label_behavioural_context_multiple_trials(datas_compiled, model_name)

    labels = [label[:, context] for label in labels]

    return get_average_input_during_context_multiple_trials(datas_compiled, labels)


def get_all_inputs_for_context(model_name, assay_config, assay_id, n, context):
    datas_compiled = []

    for i in range(1, n+1):
        datas_compiled.append(load_data(model_name, assay_config, f"{assay_id}-{i}"))

    labels = label_behavioural_context_multiple_trials(datas_compiled, model_name)

    labels = [label[:, context] for label in labels]

    labels_compiled = np.array(labels)

    compiled_observations = np.zeros((0, datas_compiled[0]["observation"].shape[1], datas_compiled[0]["observation"].shape[2],
                                      datas_compiled[0]["observation"].shape[3]))
    compiled_rnn_state = np.zeros((0, datas_compiled[0]["rnn_state_actor"].shape[1], datas_compiled[0]["rnn_state_actor"].shape[2],
                                   datas_compiled[0]["rnn_state_actor"].shape[3]))
    compiled_salt = np.zeros((0))
    compiled_energy_state = np.zeros((0))
    compiled_actions = np.zeros((0))
    compiled_in_light = np.zeros((0))

    for data, labels in zip(datas_compiled, labels_compiled):
        observations = data["observation"][labels == 1]
        rnn_state = data["rnn_state_actor"][labels == 1]
        salt = data["salt"][labels == 1]
        energy_state = data["energy_state"][labels == 1]
        actions = data["action"][labels == 1]
        in_light = data["in_light"][labels == 1]

        compiled_observations = np.concatenate((compiled_observations, observations), axis=0)
        compiled_rnn_state = np.concatenate((compiled_rnn_state, rnn_state), axis=0)
        compiled_salt = np.concatenate((compiled_salt, salt), axis=0)
        compiled_energy_state = np.concatenate((compiled_energy_state, energy_state), axis=0)
        compiled_actions = np.concatenate((compiled_actions, actions), axis=0)
        compiled_in_light = np.concatenate((compiled_in_light, in_light), axis=0)

    return compiled_observations, compiled_rnn_state, compiled_salt, compiled_energy_state, compiled_actions, \
           compiled_in_light


if __name__ == "__main__":

    x = get_mean_inputs_for_context("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10, 1)
    y = get_all_inputs_for_context("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10, 1)
