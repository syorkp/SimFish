import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    verg_angle = 77. * (np.pi / 180)
    retinal_field = 163. * (np.pi / 180)
    min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
    max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2

    angs = create_half_normal_distribution(min_angle, max_angle, 40)
    angs -= min(angs)
    angs *= 180/np.pi

    bins = np.linspace(0, max(angs)+10, 15)
    bin_diff = bins[1] - bins[0]

    density = [np.sum((angs<b+bin_diff)*(angs>=b)*1) for b in bins]
    density = np.array(density)/bin_diff
    density = list(reversed(density))

    plt.plot([i*bin_diff for i in range(len(density))], density)
    plt.ylabel("density (RFs/deg)")
    plt.xlabel("deg")
    plt.show()
