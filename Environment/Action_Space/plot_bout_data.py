import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from scipy.optimize import leastsq
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

from Environment.Action_Space.draw_angle_dist import get_pdf_for_bout
from Environment.Action_Space.draw_angle_dist import convert_action_to_bout_id


def plot_bout_data(bout_id):
    try:
        mat = scipy.io.loadmat("./Environment/Action_Space/Bout_classification/bouts.mat")
    except FileNotFoundError:
        try:
            mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")
        except FileNotFoundError:
            mat = scipy.io.loadmat("Bout_classification/bouts.mat")

    bout_id = convert_action_to_bout_id(bout_id)
    bout_id = 5

    bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
    bout_inferred_final_array = mat["BoutInfFinalArray"]

    dist_angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
    distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
    distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

    distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

    bouts = bout_inferred_final_array[:, 133].astype(int)

    relevant_bouts = (bouts == bout_id)
    dist_angles = np.absolute(dist_angles[relevant_bouts])
    distance = distance[relevant_bouts]

    dist_angles = dist_angles[distance < 15]
    dist_angles *= (np.pi/180)
    distance = distance[distance < 15]

    plt.scatter(distance, dist_angles)
    # plt.plot(np.unique(distance), np.poly1d(np.polyfit(distance, dist_angles, 1))(np.unique(distance)))
    plt.savefig("J-turn scatter (reduced outliers)")
    plt.xlabel("Dist (mm)")
    plt.ylabel("Ang (radians)")
    plt.show()
    x = True


def separate_dist_naive_bayes(action):
    try:
        mat = scipy.io.loadmat("./Environment/Action_Space/Bout_classification/bouts.mat")
    except FileNotFoundError:
        try:
            mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")
        except FileNotFoundError:
            mat = scipy.io.loadmat("Bout_classification/bouts.mat")

    bout_id = convert_action_to_bout_id(action)

    bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
    bout_inferred_final_array = mat["BoutInfFinalArray"]

    dist_angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
    distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
    distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

    distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

    bouts = bout_inferred_final_array[:, 133].astype(int)

    relevant_bouts = (bouts == bout_id)
    dist_angles = np.absolute(dist_angles[relevant_bouts])
    distance = distance[relevant_bouts]

    dist_angles *= (np.pi/180)

    mu_dist_1 = 0.4
    mu_ang_1 = 0.3

    mu_dist_2 = 0.8
    mu_ang_2 = 0.7

    mu_dist_3 = 1.2
    mu_ang_3 = 0.8

    # Generate data
    samples_1 = np.random.multivariate_normal([mu_dist_1, mu_ang_1], [[0.05, 0], [0, 0.05]], 2000)
    dist_1 = samples_1[:, 0]
    ang_1 = samples_1[:, 1]

    samples_2 = np.random.multivariate_normal([mu_dist_2, mu_ang_2], [[0.05, 0], [0, 0.05]], 2000)
    dist_2 = samples_2[:, 0]
    ang_2 = samples_2[:, 1]

    samples_3 = np.random.multivariate_normal([mu_dist_3, mu_ang_3], [[0.08, 0], [0, 0.05]], 2000)
    dist_3 = samples_3[:, 0]
    ang_3 = samples_3[:, 1]

    # plt.scatter(np.abs(dist_1), np.absolute(ang_1))
    # plt.scatter(np.abs(dist_2), np.absolute(ang_2))
    # plt.scatter(np.abs(dist_3), np.absolute(ang_3))
    # plt.show()

    samples = np.concatenate((samples_1, samples_2, samples_3), axis=0)
    values = np.concatenate((np.ones((2000)), np.ones((2000))*2, np.ones((2000))*3)).astype(int)

    classifier = GaussianNB()
    classifier.fit(samples, values)
    pred = classifier.predict(np.concatenate((np.expand_dims(distance, 1),
                                              np.expand_dims(dist_angles, 1)), axis=1))
    plt.scatter(distance, dist_angles, c=pred)
    plt.show()


if __name__ == "__main__":
    # plot_bout_data(4)
    separate_dist_naive_bayes(4)