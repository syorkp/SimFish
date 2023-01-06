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


def display_impulse_angle_space_usage(impulses, angles, figure_name, dqn=False,
                                      max_impulse=None, max_angle=None):
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


if __name__ == "__main__":
    d = load_data("dqn_beta-1", "Behavioural-Data-Free", "Naturalistic-1")
    display_binned_action_space_usage(d["action"], "Test")
    # display_impulse_angle_space_usage(impulses=d["efference_copy"][:, 0, 1],
    #                                   angles=d["efference_copy"][:, 0, 2],
    #                                   figure_name="Test",
    #                                   dqn=False,
    #                                   max_impulse=20)
    #
