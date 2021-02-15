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
                new_obs[i][j][k][0] = round(receptor[0])
                new_obs[i][j][k][1] = round(receptor[1])

    return new_obs


data = load_data("changed_penalties-1", "Naturalistic_test", "Naturalistic-1")
observation = data["observation"]
left = observation[:, :, :, 0]
left = np.swapaxes(left, 0, 1)
right = observation[:, :, :, 1]
right = np.swapaxes(right, 0, 1)

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].imshow(left, aspect="auto")
axs[1].imshow(right, aspect="auto")
plt.show()

