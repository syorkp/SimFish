import numpy as np

import matplotlib.pyplot as plt


def display_obs_sequence(obs_sequence, angles_seq=None, upscale=True):
    obs_sequence_left = np.array([o[:, 0, 0] for o in obs_sequence])
    obs_sequence_right = np.array([o[:, 0, 1] for o in obs_sequence])

    if upscale:
        max_photons = max([np.max(obs_sequence_left), np.max(obs_sequence_right)])

        obs_sequence_left = obs_sequence_left / max_photons
        obs_sequence_right = obs_sequence_right / max_photons

    if angles_seq is not None:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(obs_sequence_left)
        axs[0, 1].imshow(obs_sequence_right)
        axs[1, 1].plot(angles_seq)
        axs[0, 0].set_aspect("auto")
        axs[0, 1].set_aspect("auto")

    else:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(obs_sequence_left)
        axs[1].imshow(obs_sequence_right)
        axs[0].set_aspect("auto")
        axs[1].set_aspect("auto")

    plt.show()

