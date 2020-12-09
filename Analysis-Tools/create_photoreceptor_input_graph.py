import json
import numpy as np
import matplotlib.pyplot as plt


with open("../Assay-Output/base-1/Visual-Stimulus-Assay-1.json", "r") as file:
    data = json.load(file)

observation = [i["observation"] for i in data]

observation = np.array(observation)

new_observation = [time[0] for time in observation]

def convert_photons_to_int(obs):
    new_obs = np.zeros((121, 400, 3), int)
    for i, time in enumerate(obs):
        for j, point in enumerate(obs[i]):
            for k, receptor in enumerate(obs[i][j]):
                new_obs[i][j][k] = round(receptor)
    return new_obs


new_observation = convert_photons_to_int(new_observation)
new_observation = new_observation.transpose(1, 0, 2)
left = new_observation[:150]
right = new_observation[250:]

plt.imshow(left, aspect="auto")
plt.show()

plt.imshow(right, aspect="auto")
plt.show()


plt.show()



