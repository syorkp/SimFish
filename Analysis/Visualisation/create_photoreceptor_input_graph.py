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


def create_photoreceptor_input_graph(model, config, assay_id):
    data = load_data(model, config, assay_id)
    observation = data["observation"]
    left_1 = observation[:, :, :, 0]
    left_1 = np.swapaxes(left_1, 0, 1)
    right_1 = observation[:, :, :, 1]
    right_1 = np.swapaxes(right_1, 0, 1)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].imshow(left_1, aspect="auto")
    axs[1].imshow(right_1, aspect="auto")
    plt.show()


create_photoreceptor_input_graph("even_prey_ref-5", "Vis-Test", "Prey-Left-10")
create_photoreceptor_input_graph("even_prey_ref-5", "Vis-Test", "Naturalistic")
