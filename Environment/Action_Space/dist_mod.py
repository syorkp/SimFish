import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import scipy.io
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

from Environment.Action_Space.draw_angle_dist import convert_action_to_bout_id, get_pdf_for_bout


def narrow_dist(action, factor=3):
    normal_dist, normal_angle = get_pdf_for_bout(action)
    normal_dist[1] = normal_dist[1] ** factor
    normal_angle[1] = normal_angle[1] ** factor
    return normal_dist, normal_angle


def generate_data_mul_norm(mu_i, mu_a, sigma_i, sigma_a, n, only_positive=True):
    samples = np.random.multivariate_normal([mu_i, mu_a], [[sigma_i, 0], [0, sigma_a]], n)

    if only_positive:
        while True:
            negative_vals = samples < 0
            negative_vals = negative_vals[:, 0] | negative_vals[:, 1]
            num_negative = np.sum(negative_vals * 1)
            if num_negative > 0:
                print("Resampling")
                new_samples = np.random.multivariate_normal([mu_i, mu_a], [[sigma_i, 0], [0, sigma_a]], num_negative)
                samples[negative_vals] = new_samples
            else:
                break
    return samples


def separate_dist_naive_bayes(action):
    try:
        mat = scipy.io.loadmat("./Environment/Action_Space/Bout_classification/bouts.mat")
    except FileNotFoundError:
        try:
            mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")
        except FileNotFoundError:
            mat = scipy.io.loadmat("../../../Environment/Action_Space/Bout_classification/bouts.mat")

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


    # normal_angle[0] *= (np.pi/180)
    n_each = 2000

    mu_dist_1 = 0.4
    mu_ang_1 = 0.3
    sigma_dist_1 = 0.05
    sigma_ang_1 = 0.05

    mu_dist_2 = 0.8
    mu_ang_2 = 0.7
    sigma_dist_2 = 0.05
    sigma_ang_2 = 0.05

    mu_dist_3 = 1.2
    mu_ang_3 = 0.8
    sigma_dist_3 = 0.08
    sigma_ang_3 = 0.05

    # Generate data
    samples_1 = generate_data_mul_norm(mu_dist_1, mu_ang_1, sigma_dist_1, sigma_ang_1, n_each, True)
    samples_2 = generate_data_mul_norm(mu_dist_2, mu_ang_2, sigma_dist_2, sigma_ang_2, n_each, True)
    samples_3 = generate_data_mul_norm(mu_dist_3, mu_ang_3, sigma_dist_3, sigma_ang_3, n_each, True)

    # plt.scatter(samples_1[:, 0], samples_1[:, 1])
    # plt.scatter(samples_2[:, 0], samples_2[:, 1])
    # plt.scatter(samples_3[:, 0], samples_3[:, 1])
    # plt.show()

    samples = np.concatenate((samples_1, samples_2, samples_3), axis=0)
    values = np.concatenate((np.ones((n_each)), np.ones((n_each))*2, np.ones((n_each))*3)).astype(int)

    classifier = GaussianNB()
    classifier.fit(samples, values)
    pred = classifier.predict(np.concatenate((np.expand_dims(distance, 1),
                                              np.expand_dims(dist_angles, 1)), axis=1))
    plt.scatter(distance, dist_angles, c=pred)
    plt.show()

    # Convert to events
    # events = []
    # frequency = target[1] / np.min(target[1])
    # frequency = frequency.astype(int)
    # for i, point in enumerate(target[0]):
    #     for j in range(frequency[i]):
    #         events.append(point)
    # events = np.array(events)
    #
    # mixture = GaussianMixture(n_components=2).fit(events.reshape(-1, 1))
    # means_hat = mixture.means_.flatten()
    # weights_hat = mixture.weights_.flatten()
    # sds_hat = np.sqrt(mixture.covariances_).flatten()
    #
    # print(mixture.converged_)
    # print(means_hat)
    # print(sds_hat)
    # print(weights_hat)
    # weights_hat *= 100000
    # weights_hat = weights_hat.astype(int)
    #
    # first = np.random.normal(means_hat[0], sds_hat[0],  weights_hat[0])
    # second = np.random.normal(means_hat[1], sds_hat[1],  weights_hat[1])
    # together = np.concatenate((first, second))
    #
    # plt.hist(together, bins=1000)
    # plt.savefig("regenerated.jpg")
    # plt.show()
    #
    # plt.hist(first, bins=1000)
    # plt.savefig("regenerated-1.jpg")
    # plt.show()
    #
    # plt.hist(second, bins=1000)
    # plt.savefig("regenerated-2.jpg")
    # plt.show()


def separate_dist2(action, dist=False):
    normal_dist, normal_angle = get_pdf_for_bout(action)
    if dist:
        target = normal_dist
    else:
        target = normal_angle
        target[0] *= (np.pi/180)

    # Convert to events
    events = []
    frequency = target[1] / np.min(target[1])
    frequency = frequency.astype(int)
    for i, point in enumerate(target[0]):
        for j in range(frequency[i]):
            events.append(point)
    events = np.array(events)

    mixture = GaussianMixture(n_components=2).fit(events.reshape(-1, 1))
    means_hat = mixture.means_.flatten()
    weights_hat = mixture.weights_.flatten()
    sds_hat = np.sqrt(mixture.covariances_).flatten()

    print(mixture.converged_)
    print(means_hat)
    print(sds_hat)
    print(weights_hat)
    weights_hat *= 100000
    weights_hat = weights_hat.astype(int)

    first = np.random.normal(means_hat[0], sds_hat[0],  weights_hat[0])
    second = np.random.normal(means_hat[1], sds_hat[1],  weights_hat[1])
    together = np.concatenate((first, second))

    plt.hist(together, bins=1000)
    plt.savefig("regenerated.jpg")
    plt.show()

    plt.hist(first, bins=1000)
    plt.savefig("regenerated-1.jpg")
    plt.show()

    plt.hist(second, bins=1000)
    plt.savefig("regenerated-2.jpg")
    plt.show()


def separate_dist(action, dist=False):
    normal_dist, normal_angle = get_pdf_for_bout(action)
    if dist:
        target = normal_dist
    else:
        target = normal_angle
        target[0] *= (np.pi/180)

    # Convert to events
    events = []
    frequency = target[1] / np.min(target[1])
    frequency = frequency.astype(int)
    for i, point in enumerate(target[0]):
        for j in range(frequency[i]):
            events.append(point)

    fitfunc = lambda p, x: p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) + \
                               p[3] * np.exp(-0.5 * ((x - p[4]) / p[5]) ** 2)
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    init = [1000, 0.28, 0.1, 1000, 0.5, 0.1]

    out = leastsq(errfunc, init, args=(target[0], frequency))



    # plt.hist(events, bins=100)
    # plt.show()
    x = True


if __name__ == "__main__":
    separate_dist_naive_bayes(4)
    # j_turn_normal_dist, j_turn_normal_angle = get_pdf_for_bout(4)
    # j_turn_narrow_dist, j_turn_narrow_angle = narrow_dist(4)
    #
    # plt.plot(j_turn_normal_dist[0], j_turn_normal_dist[1] / np.sum(j_turn_normal_dist[1]))
    # plt.savefig("Normal dist")
    # plt.show()
    # plt.clf()
    #
    # plt.plot(j_turn_normal_angle[0]*(np.pi/180), j_turn_normal_angle[1] / np.sum(j_turn_normal_angle[1]))
    # plt.savefig("Normal ang")
    # plt.show()
    # plt.clf()
    #
    # plt.plot(j_turn_narrow_dist[0], j_turn_narrow_dist[1] / np.sum(j_turn_narrow_dist[1]))
    # plt.savefig("Narrow dist")
    # plt.show()
    # plt.clf()
    #
    # plt.plot(j_turn_narrow_angle[0]*(np.pi/180), j_turn_narrow_angle[1] / np.sum(j_turn_narrow_angle[1]))
    # plt.savefig("Narrow ang")
    # plt.show()
    # plt.clf()
    #
    #
    #


