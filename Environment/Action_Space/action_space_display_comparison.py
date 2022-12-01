import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import multivariate_normal
from math import log10, floor
import matplotlib.patches as mpatches

from Analysis.load_model_config import load_configuration_files_by_scaffold_point
from Environment.Action_Space.draw_angle_dist import convert_action_to_bout_id
from Environment.Action_Space.draw_angle_dist_new import draw_angle_dist_new
from Environment.Action_Space.Bout_classification.action_masking import get_action_mask


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


def get_new_bout_params(action_num):
    bout_id = convert_action_to_bout_id(action_num)

    if bout_id == 8:  # Slow2
        mean = [2.49320953e+00, 2.36217665e-19]
        cov = [[4.24434912e-01, 1.89175382e-18],
                [1.89175382e-18, 4.22367139e-03]]
    elif bout_id == 7:  # RT
        mean = [2.74619216, 0.82713249]
        cov = [[0.3839484,  0.02302918],
               [0.02302918, 0.03937928]]
    elif bout_id == 0:  # sCS
        mean = [0.956603146, -6.86735892e-18]
        cov = [[2.27928786e-02, 1.52739195e-19],
               [1.52739195e-19, 3.09720798e-03]]
    elif bout_id == 4:  # J-turn 1
        mean = [0.49074911, 0.39750791]
        cov = [[0.00679925, 0.00071446],
               [0.00071446, 0.00626601]]
    elif bout_id == 44:  # J-turn 2
        mean = [1.0535197,  0.61945679]
        # cov = [[ 0.0404599,  -0.00318193],
        #        [-0.00318193,  0.01365224]]
        cov = [[0.0404599,  0.0],
               [0.0,  0.01365224]]
    elif bout_id == 5:  # C-Start
        mean = [7.03322223, 0.67517832]
        cov = [[1.35791922, 0.10690938],
               [0.10690938, 0.10053853]]
    elif bout_id == 10:  # AS
        mean = [6.42048088e-01, 1.66490488e-17]
        cov = [[3.99909515e-02, 3.58321400e-19],
               [3.58321400e-19, 3.24366068e-03]]
    else:
        print(f"Error: {action_num}")
    return mean, cov


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
    distances, angles, labels = get_all_bouts()
    actions = ["Slow2", "RT", "sCS", "J-Turn", "C-Start", "AS"]
    # action_colours = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 0.5), (1, 1, 1)]
    action_colours = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 0.5), (0.2, 0.2, 0.2)]

    new_action_nums = [0, 1, 3, 4, 7, 9, 10]
    # new_action_colours = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 0.5), (1, 1, 1), (1, 1, 0)]
    # Inverted, for inversion operation below
    new_action_colours = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 1), (0, 1, 0.5), (0.8, 0.8, 0.8), (0, 0, 1)]

    fig, axs = plt.subplots(figsize=(10, 10))

    x = np.linspace(0, 12, num=400)
    y = np.linspace(-1, 5, num=400)
    X, Y = np.meshgrid(x, y)
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
    pdf = pdf ** 2
    pdf /= np.max(pdf)
    # pdf *= 10
    pdf -= 1
    pdf = np.absolute(pdf)

    # Add PPO action space outline
    kde_i, kde_a, ppo_action_space = get_action_mask()
    tested_region = kde_i.pdf(X.flatten()).reshape((400, 400)) * kde_a.pdf(Y.flatten()).reshape((400, 400))
    accepted_region = (tested_region > 0) * 1

    pdf[:, :, 2] += accepted_region

    axs.imshow(pdf, extent=[0, 12, -1, 5], aspect="auto")

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
    axs.set_xlim(0, 10)
    axs.set_ylim(-0.5, 2)
    plt.savefig("Plots/action_space_comparison.png")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    env, _ = load_configuration_files_by_scaffold_point("dqn_beta", 52)
    display_labelled_marques_bouts_with_dists(env)


