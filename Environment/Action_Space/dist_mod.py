import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import scipy.io
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

from Environment.Action_Space.draw_angle_dist import get_pdf_for_bout


def narrow_dist(action, factor=3):
    normal_dist, normal_angle = get_pdf_for_bout(action)
    normal_dist[1] = normal_dist[1] ** factor
    normal_angle[1] = normal_angle[1] ** factor
    return normal_dist, normal_angle


def separate_dist_naive_bayes(action):
    try:
        mat = scipy.io.loadmat("./Environment/Action_Space/Bout_classification/bouts.mat")
    except FileNotFoundError:
        try:
            mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")
        except FileNotFoundError:
            mat = scipy.io.loadmat("../../../Environment/Action_Space/Bout_classification/bouts.mat")

    normal_angle[0] *= (np.pi/180)

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
    pred = classifier.predict()

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


