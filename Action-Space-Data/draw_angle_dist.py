import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('bout_distributions.mat', 'r') as fl:
    p_angle = np.array(fl['p_angle']).T
    angles = np.array(fl['angles']).T
    p_dist = np.array(fl['p_dist']).T
    dists = np.array(fl['dists']).T
    
    
    cat = 4 # J-turn
    
    angle_cdf = np.cumsum(p_angle[cat, :]/np.sum(p_angle[cat, :]))
    dist_cdf = np.cumsum(p_dist[cat, :]/np.sum(p_dist[cat, :]))
    
    plt.figure()
    plt.subplot(211)
    plt.plot(angles[cat, :], p_angle[cat, :]/np.sum(p_angle[cat, :]))
    plt.xlabel('angle (deg)')
    plt.ylabel('PDF')
    
    plt.subplot(212)
    plt.plot(angles[cat, :], angle_cdf)
    plt.xlabel('angle (deg)')
    plt.ylabel('CDF')

    plt.figure()
    plt.subplot(211)
    plt.plot(dists[cat, :], dist_cdf)
    plt.xlabel('distance (mm)')
    plt.ylabel('PDF')
    
    plt.subplot(212)
    plt.plot(dists[cat, :], p_dist[cat, :]/np.sum(p_dist[cat, :]))
    plt.xlabel('distance (mm)')
    plt.ylabel('CDF')

    plt.show()
    
    for _ in range(10):
        r_angle = np.random.rand()
        r_dist = np.random.rand()
    
        angle_idx = np.argmin((angle_cdf - r_angle)**2)
        dist_idx = np.argmin((dist_cdf - r_dist)**2)
        
        chosen_angle = angles[cat, angle_idx]
        chosen_dist = dists[cat, dist_idx]
        
        print(f"Chosen angle: {chosen_angle}, Chosen Distance: {chosen_dist}")
