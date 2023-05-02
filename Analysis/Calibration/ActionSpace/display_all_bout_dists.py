import numpy as np
import matplotlib.pyplot as plt
import h5py


def display_bout_dist(bout_name):
    bout_id = get_marques_bout_index(bout_name) - 1

    with h5py.File('./bout_distributions.mat', 'r') as fl:
        p_angle = np.array(fl['p_angle']).T
        angles = np.array(fl['angles']).T
        p_dist = np.array(fl['p_dist']).T
        dists = np.array(fl['dists']).T

    p_angle = p_angle[bout_id]
    angles = angles[bout_id]
    p_dist = p_dist[bout_id]
    dists = dists[bout_id]

    # Make radians
    angles *= (np.pi/180)

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
    axs[0].plot(dists, p_dist/np.sum(p_dist))
    axs[0].set_xlabel("Distance (mm)")
    axs[0].set_ylabel("PDF")

    axs[1].plot(angles, p_angle/np.sum(p_angle))
    axs[1].set_xlabel("Angle (radians)")
    plt.title(bout_name)
    plt.savefig("Spatial-Density-Fish-Prey-Position-Metrics/" + bout_name, bbox_inches='tight')
    plt.clf()
    plt.close()


def get_marques_bout_index(bout_name):
    if bout_name == "Slow2":
        return 9
    elif bout_name == "RT":
        return 8
    elif bout_name == "sCS":
        return 1
    elif bout_name == "J-turn":
        return 5
    elif bout_name == "C-start":
        return 6
    elif bout_name == "AS":
        return 11
    else:
        print("ERROR - Wrong bout specified")


if __name__ == "__main__":
    bouts = ["Slow2", "RT", "sCS", "J-turn", "C-start", "AS"]
    for bout in bouts:
        display_bout_dist(bout)



