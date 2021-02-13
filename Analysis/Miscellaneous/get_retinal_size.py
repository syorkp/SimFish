import numpy as np
import matplotlib.pyplot as plt
from Analysis.load_data import load_data


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

def get_retinal_size(obs_step):
    return


data = load_data("changed_penalties-1", "Controlled_Visual_Stimuli", "Random-Prey")

observation = data["observation"]

observation = np.array(observation)

new_observation = [time[0] for time in observation]

new_observation = convert_photons_to_int(new_observation)
new_observation = new_observation.transpose(1, 0, 2)
left = new_observation[:150]
right = new_observation[150:]
