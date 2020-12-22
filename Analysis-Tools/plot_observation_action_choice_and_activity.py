import json
import numpy as np
from matplotlib import pyplot as plt
from load_data import load_data


def plot_all(rnn_data, observation, action_choice):
    fig, axs = plt.subplots(6, 1, sharex=True)

    new_observation = convert_photons_to_int(observation)
    new_observation = new_observation.transpose(1, 0, 2)
    left = new_observation[:150]
    right = new_observation[250:]

    separated_actions = []
    for action in range(7):
        action_timestamps = [i for i, a in enumerate(action_choice) if a == action]
        if len(action_timestamps) > 0:
            separated_actions.append(action_timestamps)

    colorCodes = np.array([(0.0, 0.0, 1.0),
                           (0.0, 0.75, 0.75),
                           (0.75, 0.75, 0.0),
                           (1.0, 0.0, 0.0),])

    axs[0].imshow(left, aspect="auto")
    axs[1].imshow(right, aspect="auto")
    axs[0].set_ylabel("Left Eye", fontsize=20)
    axs[1].set_ylabel("Right Eye", fontsize=20)
    axs[2].eventplot(separated_actions, color=colorCodes)
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
    new_obs = np.zeros((121, 400, 3), int)
    for i, time in enumerate(obs):
        for j, point in enumerate(obs[i]):
            for k, receptor in enumerate(obs[i][j]):
                new_obs[i][j][k] = round(receptor)
    return new_obs


data = load_data("Prey Stimuli", "Visual-Stimulus-Assay-2")

# rnn_unit_1 = [i["rnn state"][0][0] for i in data]
# rnn_unit_2 = [i["rnn state"][0][1] for i in data]
# rnn_unit_3 = [i["rnn state"][0][2] for i in data]
# rnn_unit_200 = [i["rnn state"][0][199] for i in data]

rnn_unit_1 = [data["rnn state"][i-1][0][0] for i in data["step"]]
rnn_unit_3 = [data["rnn state"][i-1][0][19] for i in data["step"]]
rnn_unit_200 = [data["rnn state"][i-1][0][190] for i in data["step"]]


action_choice = data["behavioural choice"]
unit_activity = [rnn_unit_1, rnn_unit_3, rnn_unit_200]


observation = data["observation"]

observation = np.array(observation)

new_observation = [time[0] for time in observation]

plot_all(unit_activity, new_observation, action_choice)
