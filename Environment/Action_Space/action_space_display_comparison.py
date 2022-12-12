import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import multivariate_normal
from math import log10, floor
import matplotlib.patches as mpatches

from Analysis.load_model_config import load_configuration_files_by_scaffold_point
from Environment.Action_Space.draw_angle_dist import convert_action_to_bout_id
from Environment.Action_Space.draw_angle_dist_new import draw_angle_dist_new
from Environment.Action_Space.Bout_classification.action_masking import produce_action_mask, get_new_bout_params

def calculate_energy_cost(env_variables, impulse, angle):
    """Updates the current energy state for continuous and discrete fish."""
    if env_variables["action_energy_use_scaling"] == "Nonlinear":
        unscaled_energy_use = env_variables["ci"] * (abs(impulse) ** 2) + env_variables["ca"] * (abs(angle) ** 2)
    elif env_variables["action_energy_use_scaling"] == "Linear":
        unscaled_energy_use = env_variables["ci"] * (abs(impulse)) + env_variables["ca"] * (abs(angle))
    elif env_variables["action_energy_use_scaling"] == "Sublinear":
        unscaled_energy_use = env_variables["ci"] * (abs(impulse) ** 0.5) + env_variables["ca"] * (abs(angle) ** 0.5)
    else:
        unscaled_energy_use = env_variables["ci"] * (abs(impulse) ** 0.5) + env_variables["ca"] * (abs(angle) ** 0.5)
    return unscaled_energy_use


def get_all_bouts():
    """Returns labelled distance-angle pairs of used action space"""

    action_nums = [0, 1, 3, 4, 7, 9]

    compiled_distances = []
    compiled_angles = []
    compiled_labels = []

    for action_num in action_nums:

        bout_id = convert_action_to_bout_id(action_num)
        bout_id += 1

        try:
            mat = scipy.io.loadmat("./Environment/Action_Space/Bout_classification/bouts.mat")
        except FileNotFoundError:
            try:
                mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")
            except FileNotFoundError:
                mat = scipy.io.loadmat("../../../Environment/Action_Space/Bout_classification/bouts.mat")

        bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
        bout_inferred_final_array = mat["BoutInfFinalArray"]

        angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
        distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
        distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

        distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

        bouts = bout_inferred_final_array[:, 133].astype(int)

        relevant_bouts = (bouts == bout_id)
        angles = np.absolute(angles[relevant_bouts])
        distance = distance[relevant_bouts]

        angles *= (np.pi/180)

        labels = np.array([action_num for i in range(len(angles))])

        compiled_distances.append(distance)
        compiled_angles.append(angles)
        compiled_labels.append(labels)

    compiled_distances, compiled_angles, compiled_labels = np.concatenate(compiled_distances), \
                                                           np.concatenate(compiled_angles), \
                                                           np.concatenate(compiled_labels)

    return compiled_distances, compiled_angles, compiled_labels


def display_labelled_marques_bouts():
    distances, angles, labels = get_all_bouts()
    actions = ["Slow2", "RT", "sCS", "J-Turn", "C-Start", "AS"]

    fig, axs = plt.subplots(figsize=(10, 10))
    for i, l in enumerate(set(labels)):
        axs.scatter(distances[labels == l], angles[labels == l])
    axs.legend(actions)
    axs.set_xlabel("Distance (mm)")
    axs.set_ylabel("Angle (radians)")
    axs.set_xlim(0, 12)
    axs.set_ylim(0, 5)
    plt.show()


def display_labelled_marques_bouts_with_dists(env_variables):
    res = 400
    dis_lim = [0, 15]
    ang_lim = [-1, 4]


    distances, angles, labels = get_all_bouts()
    actions = ["Slow2", "RT", "sCS", "J-Turn", "C-Start", "AS"]
    # action_colours = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 0.5), (1, 1, 1)]
    action_colours = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 0.5), (0.2, 0.2, 0.2)]

    new_action_nums = [0, 1, 3, 4, 7, 9, 10]
    # new_action_colours = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 0.5), (1, 1, 1), (1, 1, 0)]
    # Inverted, for inversion operation below
    new_action_colours = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 1), (0, 1, 0.5), (0.8, 0.8, 0.8), (0, 0, 1)]

    fig, axs = plt.subplots(figsize=(10, 10))

    impulse_range = np.linspace(dis_lim[0], dis_lim[1], res)
    angle_range = np.linspace(ang_lim[0], ang_lim[1], res)
    X, Y = np.meshgrid(impulse_range, angle_range)
    X_, Y_ = np.expand_dims(X, 2), np.expand_dims(Y, 2)
    full_grid = np.concatenate((X_, Y_), axis=2).reshape((-1, 2))
    full_grid = np.swapaxes(full_grid, 0, 1)

    pdf = np.zeros((X.shape[0], X.shape[1], 3))

    for a_n, a in enumerate(new_action_nums):
        pdf_sub = np.zeros(X.shape)
        mean, cov = get_new_bout_params(a)

        distr = multivariate_normal(cov=cov, mean=mean)
        # Generating the density function
        # for each point in the meshgrid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf_sub[i, j] += distr.pdf([X[i, j], Y[i, j]])

        # pdf_sub /= np.sum(pdf_sub)
        pdf_sub /= np.max(pdf_sub)
        # Divide by max instead, then below, sqr it as required.
        #pdf_sub *= np.sum(pdf_sub > 0) ** 2.5

        coloured_dist = np.repeat(np.expand_dims(pdf_sub, 2), 3, 2)
        coloured_dist *= new_action_colours[a_n]
        pdf += coloured_dist

        impulse = (mean[0] * 10) * 0.34452532909386484   # KEEP UPDATED WITH CALIBRATIONS
        energy_cost = calculate_energy_cost(env_variables, impulse, mean[1]) * 10e6
        # energy_cost = round(energy_cost, 2-int(floor(log10(abs(energy_cost))))-1)
        energy_cost = round(energy_cost,)
        axs.text(mean[0], mean[1], str(energy_cost), color="black", ha='center', fontsize=8)#new_action_colours[a_n])
        # axs.scatter(mean[0], mean[1], marker="x", s=100, color=[new_action_colours[a_n]])

    pdf = np.flip(pdf, 0)
    # axs.contourf(x, y, pdf, cmap='OrRd',)
    # pdf = pdf ** 2
    pdf /= np.max(pdf)
    # pdf *= 10
    pdf -= 1
    pdf = np.absolute(pdf)

    # Add PPO action space outline
    kde, threshold = produce_action_mask()
    full_grid[0] *= 3.4452532909386484  # Convert to impulses.
    tested_region = kde(full_grid).reshape((res, res))
    accepted_region = (tested_region >= threshold) * 0.2
    accepted_region = np.flip(accepted_region, 0)

    # If necessary, reflect accepted region
    if ang_lim[0] < 0:
        index_at_zero = int(res * abs(ang_lim[0])/(ang_lim[1] - ang_lim[0]))
        upper_region = accepted_region[-index_at_zero*2:-index_at_zero, :]
        accepted_region[-index_at_zero:, :] = np.flip(upper_region, 0)

    pdf -= np.expand_dims(accepted_region, 2)
    pdf = np.clip(pdf, 0, 1)

    axs.imshow(pdf, extent=[dis_lim[0], dis_lim[1], ang_lim[0], ang_lim[1]], aspect="auto")

    for i, l in enumerate(set(labels)):
        axs.scatter(distances[labels == l], angles[labels == l], alpha=0.1, color=action_colours[i], marker="x")
        axs.scatter(distances[labels == l], -angles[labels == l], alpha=0.1, color=action_colours[i], marker="x")

    # Build legend
    slow2_legend = mpatches.Patch(color=action_colours[0], label='Slow2')
    rt_legend = mpatches.Patch(color=action_colours[1], label='RT')
    sCS_legend = mpatches.Patch(color=action_colours[2], label='sCS')
    j_turn_legend = mpatches.Patch(color=action_colours[3], label='J-turn')
    c_start_legend = mpatches.Patch(color=action_colours[4], label='C-Start')
    as_legend = mpatches.Patch(color=action_colours[5], label='AS')

    axs.legend(handles=[slow2_legend, rt_legend, sCS_legend, j_turn_legend, c_start_legend, as_legend,])

    axs.set_xlabel("Distance (mm)")
    axs.set_ylabel("Angle (radians)")
    axs.set_xlim(dis_lim[0], dis_lim[1])
    axs.set_ylim(ang_lim[0], ang_lim[1])
    plt.savefig("../../Analysis-Output/Action-Space/action_space_comparison.png")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    env, _ = load_configuration_files_by_scaffold_point("dqn_beta", 52)
    display_labelled_marques_bouts_with_dists(env)


