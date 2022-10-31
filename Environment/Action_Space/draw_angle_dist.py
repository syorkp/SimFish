import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def get_pdf_for_bout(bout_id):
    data_id = convert_action_to_bout_id(bout_id)
    with h5py.File('./bout_distributions.mat', 'r') as fl:
        p_angle = np.array(fl['p_angle']).T
        angles = np.array(fl['angles']).T
        p_dist = np.array(fl['p_dist']).T
        dists = np.array(fl['dists']).T
    return [dists[data_id], p_dist[data_id]], [angles[data_id], p_angle[data_id]]


def plot_full_scatter_of_poss_actions():
    with h5py.File('./bout_distributions.mat', 'r') as fl:
        p_angle = np.array(fl['p_angle']).T
        angles = np.array(fl['angles']).T
        p_dist = np.array(fl['p_dist']).T
        dists = np.array(fl['dists']).T

        all_p_angle = p_angle.flatten()
        all_angles = angles.flatten()
        all_p_dist = p_dist.flatten()
        all_dists = dists.flatten()

        plt.scatter(all_angles, all_dists)
        plt.show()


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
        plt.plot(angles[bout_id, :]*(np.pi/180), p_angle[bout_id, :] / np.sum(p_angle[bout_id, :]))
        plt.xlabel('Angle (radians)')
        plt.ylabel('Probability Density')
        plt.savefig(f"Angle draw-{bout_id}")
        plt.clf()
        plt.close()

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
        plt.savefig(f"Distance draw-{bout_id}")
        plt.clf()
        plt.close()


def convert_action_to_bout_id(action):
    if action == 0:
        return 8
    elif action == 1 or action == 2:
        return 7
    elif action == 3:
        return 0
    elif action == 4 or action == 5:
        return 4
    elif action == 7 or action == 8:
        return 5
    elif action == 9:
        return 10


def get_modal_impulse_and_angle(action):
    """Returns the modal impulse and angle values for a specified bout, where angle is given in radians."""
    if action == 6:
        return 0.0, 0.0
    bout_id = convert_action_to_bout_id(action)

    try:
        with h5py.File('../../../Environment/Action_Space/bout_distributions.mat', 'r') as fl:
            p_angle = np.array(fl['p_angle']).T[bout_id, :]
            angles = np.array(fl['angles']).T[bout_id, :]
            p_dist = np.array(fl['p_dist']).T[bout_id, :]
            dists = np.array(fl['dists']).T[bout_id, :]

    except OSError:
        try:
            with h5py.File('../../Environment/Action_Space/bout_distributions.mat', 'r') as fl:
                p_angle = np.array(fl['p_angle']).T[bout_id, :]
                angles = np.array(fl['angles']).T[bout_id, :]
                p_dist = np.array(fl['p_dist']).T[bout_id, :]
                dists = np.array(fl['dists']).T[bout_id, :]
        except OSError:
            try:
                with h5py.File('../Environment/Action_Space/bout_distributions.mat', 'r') as fl:
                    p_angle = np.array(fl['p_angle']).T[bout_id, :]
                    angles = np.array(fl['angles']).T[bout_id, :]
                    p_dist = np.array(fl['p_dist']).T[bout_id, :]
                    dists = np.array(fl['dists']).T[bout_id, :]
            except OSError:
                with h5py.File('./Environment/Action_Space/bout_distributions.mat', 'r') as fl:
                    p_angle = np.array(fl['p_angle']).T[bout_id, :]
                    angles = np.array(fl['angles']).T[bout_id, :]
                    p_dist = np.array(fl['p_dist']).T[bout_id, :]
                    dists = np.array(fl['dists']).T[bout_id, :]

    # Get modal of both
    angle = angles[np.argmax(p_angle)]
    dist = dists[np.argmax(p_dist)]

    # Convert dist to impulse
    # impulse = (dist * 10 - (0.004644 * 140.0 + 0.081417)) / 1.771548
    impulse = (dist * 10) * 0.360574383
    angle *= (np.pi/180)

    return impulse, angle


def draw_angle_dist(bout_id):

    try:
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

    except OSError:
        try:
            with h5py.File('../../Environment/Action_Space/bout_distributions.mat', 'r') as fl:
                # with h5py.File('./bout_distributions.mat', 'r') as fl:

                p_angle = np.array(fl['p_angle']).T
                angles = np.array(fl['angles']).T
                p_dist = np.array(fl['p_dist']).T
                dists = np.array(fl['dists']).T

                angle_cdf = np.cumsum(p_angle[bout_id, :] / np.sum(p_angle[bout_id, :]))
                dist_cdf = np.cumsum(p_dist[bout_id, :] / np.sum(p_dist[bout_id, :]))

                r_angle = np.random.rand()
                r_dist = np.random.rand()

                angle_idx = np.argmin((angle_cdf - r_angle) ** 2)
                dist_idx = np.argmin((dist_cdf - r_dist) ** 2)

                chosen_angle = angles[bout_id, angle_idx]
                chosen_dist = dists[bout_id, dist_idx]
                chosen_angle = np.radians(chosen_angle)

                return chosen_angle, chosen_dist
        except OSError:
            with h5py.File('../../../Environment/Action_Space/bout_distributions.mat', 'r') as fl:
                # with h5py.File('./bout_distributions.mat', 'r') as fl:

                p_angle = np.array(fl['p_angle']).T
                angles = np.array(fl['angles']).T
                p_dist = np.array(fl['p_dist']).T
                dists = np.array(fl['dists']).T

                angle_cdf = np.cumsum(p_angle[bout_id, :] / np.sum(p_angle[bout_id, :]))
                dist_cdf = np.cumsum(p_dist[bout_id, :] / np.sum(p_dist[bout_id, :]))

                r_angle = np.random.rand()
                r_dist = np.random.rand()

                angle_idx = np.argmin((angle_cdf - r_angle) ** 2)
                dist_idx = np.argmin((dist_cdf - r_dist) ** 2)

                chosen_angle = angles[bout_id, angle_idx]
                chosen_dist = dists[bout_id, dist_idx]
                chosen_angle = np.radians(chosen_angle)

                return chosen_angle, chosen_dist


def draw_angle_dist_narrowed(bout_id, n=3):

    try:
        with h5py.File('./Environment/Action_Space/bout_distributions.mat', 'r') as fl:
        # with h5py.File('./bout_distributions.mat', 'r') as fl:

            p_angle = np.array(fl['p_angle']).T
            angles = np.array(fl['angles']).T
            p_dist = np.array(fl['p_dist']).T
            dists = np.array(fl['dists']).T
    except OSError:
        try:
            with h5py.File('../../Environment/Action_Space/bout_distributions.mat', 'r') as fl:
                # with h5py.File('./bout_distributions.mat', 'r') as fl:
                p_angle = np.array(fl['p_angle']).T
                angles = np.array(fl['angles']).T
                p_dist = np.array(fl['p_dist']).T
                dists = np.array(fl['dists']).T

        except OSError:
            with h5py.File('../../../Environment/Action_Space/bout_distributions.mat', 'r') as fl:
                # with h5py.File('./bout_distributions.mat', 'r') as fl:

                p_angle = np.array(fl['p_angle']).T
                angles = np.array(fl['angles']).T
                p_dist = np.array(fl['p_dist']).T
                dists = np.array(fl['dists']).T

    p_angle = p_angle ** n
    p_dist = p_dist ** n

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


if __name__ == "__main__":
# display_pdf_and_cdf(0)
    x, i=   draw_angle_dist(0)
    # i, a = get_modal_impulse_and_angle(7)

    # for i in range(0, 13):
    #     display_pdf_and_cdf(i)
    #     print(i)
        # angle, dist = draw_angle_dist(i)
        # print(angle, dist)

    # plot_full_scatter_of_poss_actions()
