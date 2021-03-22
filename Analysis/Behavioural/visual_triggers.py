import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import scipy.spatial.distance as dist
from scipy import stats
import numpy as np

from Analysis.load_data import load_data
from Analysis.Processing.remove_near_wall_data import remove_near_wall_data

"""
Tools to display the average visual input received when: A) A specific bout is chosen, B) A specific behavioural sequence is initiated.
"""


def convert_photons_to_int(obs):
    obs = np.array(obs)
    new_obs = np.zeros(obs.shape, int)
    for j, point in enumerate(obs):
        for k, receptor in enumerate(obs[j]):
            new_obs[j][k][0] = round(receptor[0])
            new_obs[j][k][1] = round(receptor[1])

    return new_obs


def take_observation_mean(observation_list):
    av = np.zeros(observation_list[0].shape)
    for i in observation_list:
        av = np.add(av, i)
    l = len(observation_list)
    average = np.true_divide(av, l, where=(av != 0) | (l != 0))
    average = convert_photons_to_int(average)
    return average


def take_observation_mode(observation_list):
    tally = np.zeros((120, 3, 2, len(observation_list)))
    for i, o in enumerate(observation_list):
        tally[:, :, :, i] = o
    mode = np.zeros((120, 3, 2))
    for pos in range(mode.shape[0]):
        for ph in range(mode.shape[1]):
            for eye in range(mode.shape[2]):
                mode[pos, ph, eye] = stats.mode(tally[pos, ph, eye, :]).mode
    return mode


def average_visual_input_for_bouts(p1, p2, p3, n):
    for action_num in range(0, 10):
        observation_tally = []
        for i in range(1, n + 1):
            data = load_data(p1, p2, f"{p3}-{i}")
            data = remove_near_wall_data(data, 1500, 1500)
            observation = data["observation"]
            time_points_for_bout = [i for i, a in enumerate(data["behavioural choice"]) if a == action_num]
            for index, o in enumerate(observation):
                if index in time_points_for_bout:
                    observation_tally.append(o)

        if len(observation_tally) > 0:
            average_observation = take_observation_mean(observation_tally)
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


def order_observation_tally(observation_tally):
    similarity_matrix = np.full((len(observation_tally), len(observation_tally)), 999999999)
    for i, o1 in enumerate(observation_tally):
        o1 = o1.flatten()
        for j, o2 in enumerate(observation_tally):
            if i == j:
                break
            o2 = o2.flatten()
            similarity_matrix[i, j] = dist.euclidean(o1, o2)
    ordered_observation_tally = []

    simil = divmod(similarity_matrix.argmin(), similarity_matrix.shape[1])
    ordered_observation_tally.append(observation_tally[simil[0]])
    similarity_matrix[simil[0], simil[1]] = 999999999
    current_axis = 0
    while len(ordered_observation_tally) < len(observation_tally):
        if current_axis == 0:
            current_axis = 1
        else:
            current_axis = 0
        working_simil_matrix = similarity_matrix.copy()
        for col in range(working_simil_matrix.shape[0]):
            if col == simil[current_axis]:
                pass
            else:
                for row in range(working_simil_matrix.shape[1]):
                    if row == simil[current_axis]:
                        pass
                    else:
                        working_simil_matrix[row, col] = 999999999
        # working_simil_matrix[simil[0], simil[0]] = 999999999
        similarity_matrix[simil[current_axis], :] = 999999999
        similarity_matrix[:, simil[current_axis]] = 999999999
        if np.amin(working_simil_matrix) == 999999999:
            simil = divmod(similarity_matrix.argmin(), similarity_matrix.shape[0])
        else:
            simil = divmod(working_simil_matrix.argmin(), working_simil_matrix.shape[1])
        ordered_observation_tally.append(observation_tally[simil[current_axis]])
    return ordered_observation_tally


def show_all_observations(p1, p2, p3, n):
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
            observation_tally = observation_tally[:30]
            observation_tally = order_observation_tally(observation_tally)

            fig, axs = plt.subplots(len(observation_tally), 2, sharex=True)
            fig.set_size_inches(18.5, 20)

            plt.title(f"Action: {action_num}")
            for i, o in enumerate(observation_tally):
                o = convert_photons_to_int(o)
                left = o[:, :, 0]
                right = o[:, :, 1]
                left = np.expand_dims(left, axis=0)
                right = np.expand_dims(right, axis=0)

                axs[i][0].imshow(left, aspect="auto")
                axs[i][1].imshow(right, aspect="auto")
            axs[len(observation_tally) - 1][0].set_yticklabels([])
            axs[len(observation_tally) - 1][1].set_yticklabels([])

            plt.show()


# data = load_data("changed_penalties-1", "Naturalistic_test", "Naturalistic-1")
# average_visual_input_for_bouts("changed_penalties-1", "Naturalistic", "Naturalistic", 2)
# average_visual_input_for_bouts("even_prey-1", "Naturalistic", "Naturalistic", 2)
average_visual_input_for_bouts("even_prey_ref-4", "Naturalistic", "Naturalistic", 1)
# average_visual_input_for_bouts("even_prey-1", "Predator", "Predator", 4)
# show_all_observations("changed_penalties-1", "Naturalistic", "Naturalistic", 1)
# TODO: FIx bug, some ordered observations are repeated.

x = True
