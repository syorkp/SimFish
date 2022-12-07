import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from Analysis.load_data import load_data


def plot_activity(rnn_data, observation):
    fig, axs = plt.subplots(2, 1, sharex=True)

    new_observation = convert_photons_to_int(observation)
    new_observation = new_observation.transpose(1, 0, 2)
    left = new_observation[:150]
    right = new_observation[250:]

    axs[0].imshow(left, aspect="auto")
    axs[1].imshow(right, aspect="auto")
    axs[0].set_ylabel("Left Photons")
    axs[1].set_ylabel("Right Photons")


    # axs[2].plot(rnn_data[0])
    # axs[2].set_ylabel("Unit 1 activity", fontsize=25)
    # axs[3].plot(rnn_data[1])
    # axs[3].set_ylabel("Unit 2 activity", fontsize=25)
    # axs[4].plot(rnn_data[2])
    # axs[4].set_ylabel("Unit 3 activity", fontsize=25)
    # axs[5].plot(rnn_data[3])
    # axs[5].set_ylabel("Unit 4 activity", fontsize=25)
    # axs[5].set_xlabel("Step", fontsize=25)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    # axs[2].tick_params(labelsize=15)
    # axs[3].tick_params(labelsize=15)
    # axs[4].tick_params(labelsize=15)
    # axs[5].tick_params(labelsize=15)

    # axs[0].set_ylim(0.5, 1.5)
    # fig.set_size_inches(18.5, 20)
    fig.savefig('test2png.png', dpi=100)
    plt.show()


def convert_photons_to_int(obs):
    new_obs = np.zeros((121, 400, 3), int)
    for i, time in enumerate(obs):
        for j, point in enumerate(obs[i]):
            for k, receptor in enumerate(obs[i][j]):
                new_obs[i][j][k] = round(receptor)
    return new_obs


# TODO: Make sure data load method fits.
data = load_data("changed_penalties-2", "Naturalistic", "Naturalistic-1")

rnn_unit_1 = [data["rnn state"][i-1][0][0] for i in data["step"]]
rnn_unit_2 = [data["rnn state"][i-1][0][2] for i in data["step"]]
rnn_unit_3 = [data["rnn state"][i-1][0][19] for i in data["step"]]
rnn_unit_200 = [data["rnn state"][i-1][0][190] for i in data["step"]]

observation = data["observation"]

observation = np.array(observation)

new_observation = [time[0] for time in observation]

unit_activity = [rnn_unit_1, rnn_unit_2, rnn_unit_3, rnn_unit_200]
plot_activity(unit_activity, new_observation)
