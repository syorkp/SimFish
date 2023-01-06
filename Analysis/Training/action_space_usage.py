import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from Analysis.Training.tools import find_nearest
from Analysis.load_data import load_data
from Environment.Action_Space.Bout_classification.action_masking import produce_action_mask, get_new_bout_params,\
    produce_action_mask_version_3
from Analysis.Behavioural.VisTools.get_action_name import get_action_name

"""
For creation of displays of action space usage across training.
"""


def get_impulse_angle_space_usage(impulses, angles, dqn, max_impulse, max_angle):
    res = 100    # TODO: Determine bin width as function of sample number
    angles = np.absolute(angles)

    impulse_lim = [0, 45]
    ang_lim = [0, 2]
    new_action_nums = [0, 1, 3, 4, 7, 9, 10]

    impulse_range = np.linspace(impulse_lim[0], impulse_lim[1], res)
    angle_range = np.linspace(ang_lim[0], ang_lim[1], res)
    X, Y = np.meshgrid(impulse_range, angle_range)
    X_, Y_ = np.expand_dims(X, 2), np.expand_dims(Y, 2)
    full_grid = np.concatenate((X_, Y_), axis=2).reshape((-1, 2))
    full_grid = np.swapaxes(full_grid, 0, 1)

    heatmap_array = np.zeros((X.shape[0], X.shape[1], 3))

    # Create possible usage of space in white over black
    if dqn:
        # Create whitespace of available actions
        for a_n, a in enumerate(new_action_nums):
            pdf_sub = np.zeros(X.shape)
            mean, cov = get_new_bout_params(a)

            # Convert to impulse from distance
            mean[0] *= 3.4452532909386484
            cov[0][0] *= 3.4452532909386484
            cov[0][1] *= 3.4452532909386484

            distr = multivariate_normal(cov=cov, mean=mean)
            # Generating the density function
            # for each point in the meshgrid
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pdf_sub[i, j] += distr.pdf([X[i, j], Y[i, j]])

            pdf_sub /= np.max(pdf_sub)

            coloured_dist = np.repeat(np.expand_dims(pdf_sub, 2), 3, 2)

            heatmap_array += coloured_dist

        heatmap_array[heatmap_array[:, :, 0] > 0.0001] = 1
    else:
        # Add PPO action space outline
        kde, threshold = produce_action_mask_version_3()
        full_grid[0] /= 3.4452532909386484  # Convert to distance.
        tested_region = kde(full_grid).reshape((res, res))
        accepted_region = (tested_region >= threshold) * 1
        # accepted_region = np.flip(accepted_region, 0)
        accepted_region = np.expand_dims(accepted_region, 2)

        heatmap_array[:, :, :] += accepted_region

        if max_impulse is not None:
            max_impulse_index = find_nearest(impulse_range, max_impulse)
            heatmap_array[:, max_impulse_index+1:] *= 0
        if max_angle is not None:
            max_angle_index = find_nearest(angle_range, max_angle)
            heatmap_array[max_angle_index+1:, :] *= 0


    # Overlay action choices in colours in heatmap
    nearest_i = np.array([find_nearest(impulse_range, i) for i in impulses])
    nearest_a = np.array([find_nearest(angle_range, a) for a in angles])

    heatmap_array[nearest_a, nearest_i] = np.array([1, 0, 0])
    heatmap_array = np.flip(heatmap_array, axis=0)

    return heatmap_array, impulse_lim, ang_lim


def display_impulse_angle_space_usage(impulses, angles, figure_name, dqn=False,
                                      max_impulse=None, max_angle=None):
    heatmap_array, impulse_lim, ang_lim = get_impulse_angle_space_usage(impulses, angles, dqn, max_impulse, max_angle)

    fig, axs = plt.subplots(figsize=(10, 10))
    axs.imshow(heatmap_array, extent=[impulse_lim[0], impulse_lim[1], ang_lim[0], ang_lim[1]], aspect="auto")
    axs.set_xlabel("Impulse")
    axs.set_ylabel("Angle (radians)")
    plt.savefig(f"../../Analysis-Output/Action-Space/used_action_space-{figure_name}.png")
    plt.clf()
    plt.close()


def display_binned_action_space_usage(actions, figure_name):
    action_nums, counts = np.unique(actions, return_counts=True)

    fig, axs = plt.subplots(figsize=(10, 10))
    axs.bar([get_action_name(a) for a in action_nums], counts)
    axs.set_xlabel("Action")
    axs.set_ylabel("Frequency")
    plt.savefig(f"../../Analysis-Output/Action-Space/used_actions_bouts-{figure_name}.png")
    plt.clf()
    plt.close()


def display_impulse_angle_space_usage_comparison(impulses_1, angles_1, impulses_2, angles_2, figure_name, dqn=False,
                                      max_impulse=None, max_angle=None):
    heatmap_array_1, impulse_lim, ang_lim = get_impulse_angle_space_usage(impulses_1, angles_1, dqn, max_impulse, max_angle)
    heatmap_array_2, impulse_lim, ang_lim = get_impulse_angle_space_usage(impulses_2, angles_2, dqn, max_impulse, max_angle)

    # Get all pixels with colour - these are the possible actions
    possible_combinations = ((np.sum(heatmap_array_1, axis=2) >= 1.) + (np.sum(heatmap_array_2, axis=2) >= 1.)) * 1.0

    # Get all pixels that are only red - these are actions used in either case
    used_actions_1 = (heatmap_array_1[:, :, 0] >= 1.) * (heatmap_array_1[:, :, 1] < 1.) * (heatmap_array_1[:, :, 2] < 1.) * 1
    used_actions_2 = (heatmap_array_2[:, :, 0] >= 1.) * (heatmap_array_2[:, :, 1] < 1.) * (heatmap_array_2[:, :, 2] < 1.) * 1

    used_actions_difference = used_actions_2 - used_actions_1

    heatmap_array = np.repeat(np.expand_dims(possible_combinations, 2), 3, 2)
    heatmap_array[used_actions_difference >= 1] = np.array([1., 0, 0])
    heatmap_array[used_actions_difference <= -1] = np.array([0, 0, 1.])

    fig, axs = plt.subplots(figsize=(10, 10))
    axs.imshow(heatmap_array, extent=[impulse_lim[0], impulse_lim[1], ang_lim[0], ang_lim[1]], aspect="auto")
    axs.set_xlabel("Impulse")
    axs.set_ylabel("Angle (radians)")
    plt.savefig(f"../../Analysis-Output/Action-Space/used_action_space-{figure_name}-comparison.png")
    plt.clf()
    plt.close()


def display_binned_action_space_usage_comparison(actions_1, actions_2, figure_name):
    action_nums_1, counts_1 = np.unique(actions_1, return_counts=True)
    action_nums_2, counts_2 = np.unique(actions_2, return_counts=True)

    fig, axs = plt.subplots(figsize=(15, 10))
    axs.bar([i-0.2 for i in action_nums_1], counts_1, width=0.4)
    axs.bar([i+0.2 for i in action_nums_2], counts_2, width=0.4)

    labels = [get_action_name(a) for a in range(12)]
    axs.set_xticks([i for i in range(len(labels))])
    axs.set_xticklabels(labels)

    axs.set_xlabel("Action")
    axs.set_ylabel("Frequency")
    plt.savefig(f"../../Analysis-Output/Action-Space/used_actions_bouts-{figure_name}-comparison.png")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    d1 = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    d2 = load_data("dqn_beta-1", "Behavioural-Data-Free", "Naturalistic-1")
    # display_impulse_angle_space_usage_comparison(impulses_1=d1["impulse"],
    #                                              angles_1=d1["angle"],
    #                                              impulses_2=d2["efference_copy"][:, 0, 1],
    #                                              angles_2=d2["efference_copy"][:, 0, 2],
    #                                              figure_name="Test",
    #                                              dqn=True)
    display_binned_action_space_usage_comparison(d1["action"], d2["action"], "Test")

    # display_binned_action_space_usage(d["action"], "Test")
    # display_impulse_angle_space_usage(impulses=d["impulse"],
    #                                   angles=d["angle"],
    #                                   figure_name="Test",
    #                                   dqn=True)
