import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

from Analysis.load_data import load_data

"""
Tools to display the average visual input received when: A) A specific bout is chosen, B) A specific behavioural sequence is initiated.
"""


def average_visual_input_for_bout_sequence(p1, p2, p3, n, bout_sequence):
    ...


def convert_photons_to_int(obs):
    obs = np.array(obs)
    new_obs = np.zeros(obs.shape, int)
    for j, point in enumerate(obs):
        for k, receptor in enumerate(obs[j]):
            new_obs[j][k][0] = round(receptor[0])
            new_obs[j][k][1] = round(receptor[1])

    return new_obs


def take_observation_average(observation_list):
    av = np.zeros(observation_list[0].shape)
    for i in observation_list:
        av = np.add(av, i)
    l = len(observation_list)
    average = np.true_divide(av, l, where=(av != 0) | (l != 0))
    average = convert_photons_to_int(average)
    return average


def average_visual_input_for_bouts(p1, p2, p3, n):
    for action_num in range(0, 10):
        observation_tally = []
        for i in range(1, n + 1):
            data = load_data(p1, p2, f"{p3}-{i}")
            observation = data["observation"]
            time_points_for_bout = [i for i, a in enumerate(data["behavioural choice"]) if a == action_num]
            for index, o in enumerate(observation):
                if index in time_points_for_bout:
                    observation_tally.append(o)

        if len(observation_tally) > 0:
            average_observation = take_observation_average(observation_tally)
            left = average_observation[:, :, 0]
            right = average_observation[:, :, 1]
            left = np.expand_dims(left, axis=0)
            right = np.expand_dims(right, axis=0)

            fig, axs = plt.subplots(2, 1, sharex=True)

            plt.title(f"Action: {action_num}")
            axs[0].imshow(left, aspect="auto")
            axs[0].set_yticklabels([])
            axs[1].imshow(right, aspect="auto")
            axs[1].set_yticklabels([])

            plt.show()
    return


data = load_data("changed_penalties-1", "Naturalistic_test", "Naturalistic-1")
average_visual_input_for_bouts("changed_penalties-1", "Naturalistic", "Naturalistic", 2)

x = True
