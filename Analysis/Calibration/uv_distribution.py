import numpy as np
import matplotlib.pyplot as plt


# Testing distribution.

def create_half_normal_distribution(min_angle, max_angle, photoreceptor_num, sigma=1):
    mu = 0
    angle_difference = abs(max_angle - min_angle)

    angle_range = np.linspace(min_angle, max_angle, photoreceptor_num)
    frequencies = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(angle_range - mu) ** 2 / (2 * sigma ** 2))
    differences = 1 / frequencies
    differences[0] = 0
    total_difference = np.sum(differences)
    differences = (differences * angle_difference) / total_difference
    cumulative_differences = np.cumsum(differences)
    photoreceptor_angles = min_angle + cumulative_differences

    # Computing Indices for resampling
    # hypothetical_angle_range = np.linspace(min_angle, max_angle, self.max_photoreceptor_num)
    # hypothetical_frequencies = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(hypothetical_angle_range-mu)**2/(2*sigma**2))
    # hypothetical_differences = 1/hypothetical_frequencies
    # hypothetical_differences[0] = 0
    # hypothetical_total_difference = np.sum(hypothetical_differences)
    # hypothetical_differences = (hypothetical_differences*angle_difference)/hypothetical_total_difference
    # hypothetical_cumulative_differences = np.cumsum(hypothetical_differences)
    # hypothetical_photoreceptor_angles = min_angle + hypothetical_cumulative_differences
    # relative_indices = np.round(((hypothetical_photoreceptor_angles - min(hypothetical_photoreceptor_angles))/angle_difference) * (photoreceptor_num-1)).astype(int)

    return photoreceptor_angles  # , relative_indices


def update_angles_strike_zone(verg_angle, retinal_field, is_left, photoreceptor_num, sigma):
    """Set the eyes visual angles, with the option of particular distributions."""

    # Half normal distribution
    if is_left:
        min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
        max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2

    else:
        min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
        max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2

    # sampled_values = self.sample_half_normal_distribution(min_angle, max_angle, photoreceptor_num)
    computed_values = create_half_normal_distribution(min_angle, max_angle, photoreceptor_num, sigma)

    plt.scatter(computed_values, [1 for i in range(len(computed_values))])
    plt.show()

    plt.hist(computed_values, bins=20)
    plt.title(f"Sigma: {sigma}")
    plt.show()
    # self.indices_for_padding_uv = relative_indices

    return computed_values

# for i in np.linspace(0.5, 2.5, 10):
#     update_angles_strike_zone(77. * (np.pi / 180), 163. * (np.pi / 180), True, 55, i)


ipl_positions = np.loadtxt("./Zimmerman/doi_10.5061_dryad.5bc8vd7__v1/Processed data/IPLPositions_GCaMP2.txt", dtype=float)[:, 0]
clusters = np.loadtxt("./Zimmerman/doi_10.5061_dryad.5bc8vd7__v1/Processed data/ClusterIndex_GCaMP2.txt", dtype=float)
cluster_class = np.loadtxt("./Zimmerman/doi_10.5061_dryad.5bc8vd7__v1/Processed data/ClusterClass2.txt", dtype=int)

with open("./Zimmerman/doi_10.5061_dryad.5bc8vd7__v1/Processed data/ClusterIndex_GCaMP2.txt") as f:
    lines = f.readlines()
    n_cols = len(f.readline().split(","))

lines = np.array([25 if line == "\n" else int(line) for line in lines]).astype(int)
uv_clusters = [2, 3]
added_clusters = np.array([cluster_class[line] for line in lines])[:6542]
# indexes_of_uv = [i if (cluster_class[line]==2) else None for i, line in enumerate(lines)]
relevant_positions = np.concatenate((ipl_positions[added_clusters == 2], ipl_positions[added_clusters == 3]), axis=0)
x = True

plt.hist(relevant_positions, bins=20)
plt.show()

plt.hist(ipl_positions, bins=20)
plt.show()

