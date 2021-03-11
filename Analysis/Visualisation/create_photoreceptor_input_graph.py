import json
import numpy as np
import matplotlib.pyplot as plt
from Analysis.load_data import load_data


def convert_photons_to_int(obs):
    # In proper format isnt necessary
    obs = np.array(obs)
    new_obs = np.zeros(obs.shape, int)
    for i, time in enumerate(obs):
        for j, point in enumerate(obs[i]):
            for k, receptor in enumerate(obs[i][j]):
                new_obs[i][j][k] = round(receptor)
                # new_obs[i][j][k][1] = round(receptor[1])

    return new_obs


data = load_data("even_prey_ref-1", "Controlled_Visual_Stimuli", "Prey-Moving-Left")
observation = data["observation"]
left_1 = observation[:, :, :, 0]
left_1 = np.swapaxes(left_1, 0, 1)
right_1 = observation[:, :, :, 1]
right_1 = np.swapaxes(right_1, 0, 1)

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].imshow(left_1, aspect="auto")
axs[1].imshow(right_1, aspect="auto")
plt.show()


observation = data["rev_observation"]
left = np.squeeze(observation[:, 0, :, :, :])
left = np.swapaxes(left, 0, 1)
right = np.squeeze(observation[:, 1, :, :, :])
right = np.swapaxes(right, 0, 1)
left = convert_photons_to_int(left)
right = convert_photons_to_int(right)

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].imshow(left, aspect="auto")
axs[1].imshow(right, aspect="auto")
plt.show()

