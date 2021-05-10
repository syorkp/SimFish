import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_pdf_and_cdf(bout_id):
    with h5py.File('./bout_distributions.mat', 'r') as fl:
        p_angle = np.array(fl['p_angle']).T
        angles = np.array(fl['angles']).T
        p_dist = np.array(fl['p_dist']).T
        dists = np.array(fl['dists']).T
        angle_cdf = np.cumsum(p_angle[bout_id, :] / np.sum(p_angle[bout_id, :]))
        dist_cdf = np.cumsum(p_dist[bout_id, :] / np.sum(p_dist[bout_id, :]))

        sns.set()
        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        # plt.xlabel('angle (deg)')
        plt.plot(angles[bout_id, :], angle_cdf)
        plt.ylabel('CDF')
        plt.title(label=bout_id+1)

        plt.subplot(212)
        plt.plot(angles[bout_id, :], p_angle[bout_id, :] / np.sum(p_angle[bout_id, :]))

        plt.xlabel('Angle (degrees)')
        plt.ylabel('Probability Density')

        plt.show()

        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        plt.plot(dists[bout_id, :], dist_cdf)
        # plt.xlabel('distance (mm)')
        plt.ylabel('CDF')
        plt.title(label=bout_id+1)

        plt.subplot(212)
        plt.plot(dists[bout_id, :], p_dist[bout_id, :] / np.sum(p_dist[bout_id, :]))
        plt.xlabel('Distance (mm)')
        plt.ylabel('Probability Density')

        plt.show()


def draw_angle_dist(bout_id):

    with h5py.File('./Environment/Action_Space/bout_distributions.mat', 'r') as fl:
# with h5py.File('./bout_distributions.mat', 'r') as fl:

        p_angle = np.array(fl['p_angle']).T
        angles = np.array(fl['angles']).T
        p_dist = np.array(fl['p_dist']).T
        dists = np.array(fl['dists']).T

        angle_cdf = np.cumsum(p_angle[bout_id, :]/np.sum(p_angle[bout_id, :]))
        dist_cdf = np.cumsum(p_dist[bout_id, :]/np.sum(p_dist[bout_id, :]))

        r_angle = np.random.rand()
        r_dist = np.random.rand()

        angle_idx = np.argmin((angle_cdf - r_angle)**2)
        dist_idx = np.argmin((dist_cdf - r_dist)**2)

        chosen_angle = angles[bout_id, angle_idx]
        chosen_dist = dists[bout_id, dist_idx]
        chosen_angle = np.radians(chosen_angle)

        return chosen_angle, chosen_dist


# for i in range(0, 13):
#     # display_pdf_and_cdf(i)
#     print(i)
#     angle, dist = draw_angle_dist(i)
#     print(angle, dist)

# display_pdf_and_cdf(5)
# display_pdf_and_cdf(9)
