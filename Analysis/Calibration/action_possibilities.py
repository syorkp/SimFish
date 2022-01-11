"""TO generate action mask for marques et al data."""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_full_scatter_of_poss_actions():
    with h5py.File('../../Environment/Action_Space/bout_distributions.mat', 'r') as fl:
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

plot_full_scatter_of_poss_actions()