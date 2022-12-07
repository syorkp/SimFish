import json
import numpy as np
from matplotlib import pyplot as plt
from Analysis.load_data import load_data
import seaborn as sns


def plot_actions_and_observations(observation, action_choice):
    fig, axs = plt.subplots(3, 1, sharex=True)

    # new_observation = convert_photons_to_int(observation)
    observation = observation.transpose(1, 0, 2, 3)
    left = observation[:, :, :, 0]
    right = observation[:, :, :, 1]

    separated_actions = []
    for action in range(max(action_choice)):
        action_timestamps = [i for i, a in enumerate(action_choice) if a == action]
        # if len(action_timestamps) > 0:
        separated_actions.append(action_timestamps)

    colors = sns.color_palette("hls", max(action_choice))

    axs[0].imshow(left, aspect="auto")
    axs[1].imshow(right, aspect="auto")
    axs[0].set_ylabel("Left Eye", fontsize=20)
    axs[1].set_ylabel("Right Eye", fontsize=20)
    axs[2].eventplot(separated_actions, color=colors)
    axs[2].set_ylabel("Action Choice", fontsize=20)
    axs[2].set_xlabel("Step", fontsize=20)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[2].tick_params(labelsize=15)

    fig.set_size_inches(15, 16)
    fig.savefig('test2png.png', dpi=100)
    plt.show()


def plot_all(rnn_data, observation, action_choice):
    fig, axs = plt.subplots(6, 1, sharex=True)

    new_observation = convert_photons_to_int(observation)
    new_observation = new_observation.transpose(1, 0, 2)
    left = new_observation[:]
    right = new_observation[400:]

    separated_actions = []
    for action in range(10):
        action_timestamps = [i for i, a in enumerate(action_choice) if a == action]
        if len(action_timestamps) > 0:
            separated_actions.append(action_timestamps)


    colorCodes = np.array([(0.0, 0.0, 1.0),
                           (0.0, 0.75, 0.75),
                           (0.75, 0.75, 0.0),
                           (1.0, 0.0, 0.0),
                           (1.0, 0.75, 0.0),
                           (0.75, 0.0, 1.0),
                           (0.5, 0.5, 0.5),
                           ])
    # TODO: Select number of codes automatically
    colors = sns.color_palette("hls", len(set(action_choice)))

    axs[0].imshow(left, aspect="auto")
    axs[1].imshow(right, aspect="auto")
    axs[0].set_ylabel("Left Eye", fontsize=20)
    axs[1].set_ylabel("Right Eye", fontsize=20)
    axs[2].eventplot(separated_actions, color=colors)
    axs[2].set_ylabel("Action Choice", fontsize=20)
    axs[3].plot(rnn_data[0])
    axs[3].set_ylabel("Unit 1 activity", fontsize=20)
    axs[4].plot(rnn_data[1])
    axs[4].set_ylabel("Unit 2 activity", fontsize=20)
    axs[5].plot(rnn_data[2])
    axs[5].set_ylabel("Unit 3 activity", fontsize=20)
    axs[5].set_xlabel("Step", fontsize=20)

    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[2].tick_params(labelsize=15)
    axs[3].tick_params(labelsize=15)
    axs[4].tick_params(labelsize=15)
    axs[5].tick_params(labelsize=15)

    fig.set_size_inches(20.5, 22)
    fig.savefig('test2png.png', dpi=100)
    plt.show()


def convert_photons_to_int(obs):
    obs = np.array(obs)

    new_obs = np.zeros(obs.shape, int)
    for i, time in enumerate(obs):
        # if i == 121:
        #     break
        for j, point in enumerate(obs[i]):
            # if j == 400:
            #     break
            for k, receptor in enumerate(obs[i][j]):
                new_obs[i][j][k] = round(receptor)
    return new_obs

#
# data = load_data("changed_penalties-1", "Naturalistic_test", "Naturalistic-1")
#
# rnn_unit_1 = [data["rnn state"][i-1][0][165] for i in data["step"]]
# rnn_unit_2 = [data["rnn state"][i-1][0][155] for i in data["step"]]
# rnn_unit_3 = [data["rnn state"][i-1][0][15] for i in data["step"]]
# rnn_unit_200 = [data["rnn state"][i-1][0][139] for i in data["step"]]
#
# # conv_unit_1 = [data["left_conv_1"][i-1][0][0] for i in data["step"]]
# # conv_unit_2 = [data["left_conv_1"][i-1][0][1] for i in data["step"]]
# # conv_unit_3 = [data["left_conv_1"][i-1][0][2] for i in data["step"]]
# # conv_unit_4 = [data["left_conv_1"][i-1][0][26] for i in data["step"]]
#
#
# action_choice = data["behavioural choice"]
# # unit_activity = [conv_unit_1, conv_unit_2, conv_unit_3, conv_unit_4]
# unit_activity = [rnn_unit_1, rnn_unit_2, rnn_unit_3, rnn_unit_200]
#
# observation = data["observation"]
#
# observation = np.array(observation)
#
# new_observation = [time[0] for time in observation]
#
# plot_all(unit_activity, new_observation, action_choice)

data = load_data("even_prey_ref-4", "Behavioural-Data-Free", "Prey-1")
plot_actions_and_observations(data["observation"][200: 500], data["behavioural choice"][200: 500])

