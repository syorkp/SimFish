import numpy as np
import json
import matplotlib.pyplot as plt
import copy

from Analysis.load_data import load_data


def load_gradients_data(model_name, target_layer, context_name, method=None):
    if method is None:
        with open(f"./Gradients-Data/{model_name}/Gradients-{target_layer}-{context_name}.json", "r") as outfile:
            gradients = json.load(outfile)
    else:
        with open(f"./Gradients-Data/{model_name}/Gradients-{target_layer}-{context_name}-{method}.json", "r") as outfile:
            gradients = json.load(outfile)
    return gradients


def load_stimuli_data(model_name, context_name):
    with open(f"./Gradients-Data/{model_name}/Stimuli-{context_name}.json", "r") as outfile:
        stimuli = json.load(outfile)
    return stimuli


def display_all_gradients(gradients, stimuli=None):
    max_grad = max([np.max(gradients[key]) for key in gradients.keys()])
    min_grad = min([np.min(gradients[key]) for key in gradients.keys()])

    observation_gradients = np.array(gradients["dY_dOBS"])
    energy_gradients = np.array(gradients["dY_dENR"])
    efference_gradients = np.array(gradients["dY_dEFF"])
    in_light_gradients = np.array(gradients["dY_dLIT"])
    salt_gradients = np.array(gradients["dY_dSLT"])
    rnn_gradients = np.array(gradients["dY_dRNN"])

    # if not isinstance(in_light_gradients, float):
    #     in_light_gradients = in_light_gradients[0]
    #     energy_gradients = energy_gradients[0]
    #     salt_gradients = salt_gradients[0]

    num_gradients = observation_gradients.shape[0]
    internal_state_gradients = np.concatenate((np.expand_dims(energy_gradients, 1),
                                               np.expand_dims(salt_gradients, 1),
                                               np.expand_dims(in_light_gradients, 1)), axis=1)

    observation_yscaling = int(0.5/(observation_gradients.shape[0] / observation_gradients.shape[1]))
    observation_gradients = np.repeat(observation_gradients, observation_yscaling, 0)
    observation_gradients = np.concatenate((observation_gradients[:, :, 0:1], observation_gradients[:, :, 2:3], observation_gradients[:, :, 1:2]), axis=2)

    positive_observation_gradients = copy.copy(observation_gradients)
    positive_observation_gradients[(positive_observation_gradients < 0)] = 0
    positive_observation_gradients = positive_observation_gradients / np.max(positive_observation_gradients)

    negative_observation_gradients = copy.copy(observation_gradients)
    negative_observation_gradients[(negative_observation_gradients > 0)] = 0
    negative_observation_gradients = negative_observation_gradients / np.min(negative_observation_gradients)

    # Display Observation and Gradients
    if stimuli is None:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(positive_observation_gradients[:, :, :, 0])
        axs[1, 0].imshow(positive_observation_gradients[:, :, :, 1])
        axs[0, 1].imshow(negative_observation_gradients[:, :, :, 0])
        axs[1, 1].imshow(negative_observation_gradients[:, :, :, 1])
    else:
        observation = np.array(stimuli["observation"]).astype(int)
        observation = np.concatenate((observation[:, 0:1, :], observation[:, 2:3], observation[:, 1:2]), axis=1)
        observation = np.expand_dims(observation, 0)
        observation = np.repeat(observation, int(observation_yscaling)*num_gradients, 0)

        fig, axs = plt.subplots(3, 2, sharex=True)
        axs[0, 0].imshow(observation[:, :, :, 0])
        axs[0, 1].imshow(observation[:, :, :, 1])

        axs[1, 0].imshow(positive_observation_gradients[:, :, :, 0])
        axs[1, 1].imshow(positive_observation_gradients[:, :, :, 1])
        axs[2, 0].imshow(negative_observation_gradients[:, :, :, 0])
        axs[2, 1].imshow(negative_observation_gradients[:, :, :, 1])
    fig.suptitle("Observation")
    plt.show()

    internal_state_yscaling = int(0.5 / (internal_state_gradients.shape[0] / internal_state_gradients.shape[1]))
    internal_state_gradients = np.repeat(internal_state_gradients, internal_state_yscaling, 0)

    # Display Internal State and Gradients
    if stimuli is None:
        plt.imshow(internal_state_gradients)
    else:
        energy_state, salt, in_light = stimuli["energy_state"], stimuli["salt"], stimuli["in_light"]

        internal_state = np.array([energy_state, salt, in_light])
        internal_state = np.expand_dims(internal_state, 0)
        internal_state = np.repeat(internal_state, internal_state_yscaling, 0)

        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(internal_state)
        axs[1].imshow(internal_state_gradients)
    fig.suptitle("Internal State")
    plt.show()

    rnn_yscaling = int(0.5/(rnn_gradients.shape[0] / rnn_gradients.shape[2]))
    rnn_gradients = np.repeat(rnn_gradients, rnn_yscaling, 0)

    # Plot RNN state and gradients
    if stimuli is None:
        fig, axs = plt.subplots(2, 1)

        axs[0].imshow(rnn_gradients[:, 0])
        axs[1].imshow(rnn_gradients[:, 1])
    else:
        rnn_state = np.array(stimuli["rnn_state"])[:, 0, :]
        rnn_state = np.expand_dims(rnn_state, 0)
        rnn_state = np.repeat(rnn_state, rnn_yscaling, 0)

        fig, axs = plt.subplots(2, 2)

        axs[0, 0].imshow(rnn_state[:, 0])
        axs[0, 1].imshow(rnn_state[:, 1])

        axs[1, 0].imshow(rnn_gradients[:, 0])
        axs[1, 1].imshow(rnn_gradients[:, 1])
    fig.suptitle("RNN State")
    plt.show()

    efference_yscaling = int(0.5/(efference_gradients.shape[0] / efference_gradients.shape[1]))
    efference_gradients = np.repeat(efference_gradients, efference_yscaling, 0)

    # Plot Efference
    if stimuli is None:
        plt.imshow(efference_gradients)
    else:
        one_hot_actions = np.zeros((efference_yscaling, 10))
        efference = np.array(stimuli["efference_copy"][1:])
        efference = np.expand_dims(efference, 0)
        efference = np.repeat(efference, efference_yscaling, 0)
        one_hot_actions[:, int(stimuli["efference_copy"][0])] = 1

        fig, axs = plt.subplots(2, 2, sharex=True)

        axs[0, 0].imshow(one_hot_actions)
        axs[0, 1].imshow(efference)
        axs[1, 0].imshow(efference_gradients)
    fig.suptitle("Efference")
    plt.show()


def normalise_all_gradients(gradients):
    observation_gradients = np.array(gradients["dY_dOBS"])
    energy_gradients = np.array(gradients["dY_dENR"])
    efference_gradients = np.array(gradients["dY_dEFF"])
    in_light_gradients = np.array(gradients["dY_dLIT"])
    salt_gradients = np.array(gradients["dY_dSLT"])
    rnn_gradients =   np.array(gradients["dY_dRNN"])

    observation_gradients_max = np.max(observation_gradients)
    energy_gradients_max = np.max(energy_gradients)
    efference_gradients_max = np.max(efference_gradients)
    in_light_gradients_max = np.max(in_light_gradients)
    salt_gradients_max = np.max(salt_gradients)
    rnn_gradients_max = np.max(rnn_gradients)
    gradients_max = max(observation_gradients_max, energy_gradients_max, efference_gradients_max,
                        in_light_gradients_max, salt_gradients_max, rnn_gradients_max)

    observation_gradients_min = np.min(observation_gradients)
    energy_gradients_min = np.min(energy_gradients)
    efference_gradients_min = np.min(efference_gradients)
    in_light_gradients_min = np.min(in_light_gradients)
    salt_gradients_min = np.min(salt_gradients)
    rnn_gradients_min = np.min(rnn_gradients)
    gradients_min = min(observation_gradients_min, energy_gradients_min, efference_gradients_min,
                        in_light_gradients_min, salt_gradients_min, rnn_gradients_min)

    # observation_gradients_sum = np.sum(observation_gradients[3])
    # energy_gradients_sum = np.sum(energy_gradients[3])
    # efference_gradients_sum = np.sum(efference_gradients[3])
    # in_light_gradients_sum = np.sum(in_light_gradients[3])
    # salt_gradients_sum = np.sum(salt_gradients[3])
    # rnn_gradients_sum = np.sum(rnn_gradients[3])

    observation_gradients[observation_gradients>0] /= gradients_max
    energy_gradients[energy_gradients>0] /= gradients_max
    efference_gradients[efference_gradients>0] /= gradients_max
    in_light_gradients[in_light_gradients>0] /= gradients_max
    salt_gradients[salt_gradients>0] /= gradients_max
    rnn_gradients[rnn_gradients>0] /= gradients_max

    observation_gradients[observation_gradients < 0] /= gradients_min
    energy_gradients[energy_gradients < 0] /= gradients_min
    efference_gradients[efference_gradients < 0] /= gradients_min
    in_light_gradients[in_light_gradients < 0] /= gradients_min
    salt_gradients[salt_gradients < 0] /= gradients_min
    rnn_gradients[rnn_gradients < 0] /= gradients_min

    new_gradients = {
        "dY_dOBS": observation_gradients,
        "dY_dENR": energy_gradients,
        "dY_dEFF": efference_gradients,
        "dY_dLIT": in_light_gradients,
        "dY_dSLT": salt_gradients,
        "dY_dRNN": rnn_gradients,
    }
    return new_gradients


if __name__ == "__main__":
    # d = load_data("dqn_scaffold_18-1", "Behavioural-Data-Test", "Naturalistic-4")
    # pc_stimuli = load_stimuli_data("dqn_scaffold_18-1", "Prey Capture")
    # pc_gradients = load_gradients_data("dqn_scaffold_18-1", "Advantage", "Prey Capture")
    # ex_stimuli = load_stimuli_data("dqn_scaffold_18-1", "Exploration")
    # ex_gradients = load_gradients_data("dqn_scaffold_18-1", "Advantage", "Exploration")
    # wi_stimuli = load_stimuli_data("dqn_scaffold_18-1", "Wall Interaction")
    # wi_gradients = load_gradients_data("dqn_scaffold_18-1", "Advantage", "Wall Interaction")
    # st_stimuli = load_stimuli_data("dqn_scaffold_18-1", "Starving")
    # st_gradients = load_gradients_data("dqn_scaffold_18-1", "Advantage", "Starving")
    # display_all_gradients(pc_gradients, pc_stimuli)
    # display_all_gradients(ex_gradients, ex_stimuli)
    # display_all_gradients(wi_gradients, wi_stimuli)
    # display_all_gradients(st_gradients, st_stimuli)

    pc_stimuli = load_stimuli_data("dqn_scaffold_18-1", "Consumption")
    pc_gradients = load_gradients_data("dqn_scaffold_18-1", "Advantage", "Consumption", "averaged")
    pc_gradients = normalise_all_gradients(pc_gradients)

    display_all_gradients(pc_gradients, pc_stimuli)
