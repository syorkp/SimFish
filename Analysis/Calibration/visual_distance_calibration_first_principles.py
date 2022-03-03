import numpy as np
import matplotlib.pyplot as plt


def get_max_scatter_photons(bkg_scatter, distance, rf_size, decay_constant):
    photons = 0

    for d in range(int(distance)):
        point_width = 2 * d * np.tan(rf_size/2)
        distance_scaling = np.exp(-decay_constant * d) * bkg_scatter
        if point_width < 1:
            point_width = 1
        photons += distance_scaling * point_width

    return photons


def get_max_prey_photons(distance, decay_constant, luminance):

    pixel_value = np.exp(-decay_constant * distance) * luminance

    return pixel_value


def compute_distinguishability(prey_stimulus, max_noise_stimulus):
    distinguishability = (prey_stimulus - max_noise_stimulus)
    return distinguishability


def plot_distinguishability_against_distance(max_distance, bkg_scatter, luminance, scaling_factor):
    rf_size = 0.73
    decay_constant = 0.006
    visual_distance = 74

    distinguishability_scores = []
    distances = np.linspace(10, max_distance, 100)

    uv_prey_photons = []
    uv_scatter_photons = []

    for distance in distances:
        max_uv_scatter = get_max_scatter_photons(bkg_scatter, max_distance, rf_size, decay_constant)
        uv_scatter_prey = get_max_scatter_photons(bkg_scatter, distance, rf_size, decay_constant)
        uv_prey = get_max_prey_photons(distance, decay_constant, luminance)
        uv_prey_stimulus = uv_prey + uv_scatter_prey

        max_uv_scatter *= scaling_factor
        uv_prey_stimulus *= scaling_factor
        max_uv_scatter = int(max_uv_scatter)
        uv_prey_stimulus = int(uv_prey_stimulus)

        distinguishability_scores.append(compute_distinguishability(uv_prey_stimulus, max_uv_scatter))

        uv_prey_photons.append(uv_prey_stimulus)
        uv_scatter_photons.append(max_uv_scatter)

    distinguishability_range = 1.96 * np.sqrt(np.abs(np.array(distinguishability_scores)))
    plt.plot(distances, distinguishability_scores)
    plt.title(f"BKG: {bkg_scatter}, Luminance: {luminance}")
    plt.vlines(visual_distance, min(distinguishability_scores), max(distinguishability_scores), color="r")
    plt.fill_between(distances, np.array(distinguishability_scores)-distinguishability_range,
                     np.array(distinguishability_scores)+distinguishability_range, alpha=0.2)
    plt.show()

    # Absolute photon values
    range_uv_prey = 1.96 * np.sqrt(np.abs(np.array(uv_prey_photons)))
    range_uv_scatter = 1.96 * np.sqrt(np.abs(np.array(uv_scatter_photons)))
    plt.plot(distances, uv_prey_photons, color="r")
    plt.plot(distances, uv_scatter_photons, color="b")
    plt.title(f"Absolute photon values. BKG: {bkg_scatter}, Luminance: {luminance}")
    plt.vlines(visual_distance, min(distinguishability_scores), max(uv_prey_photons) + max(range_uv_prey), color="green")
    plt.fill_between(distances, np.array(uv_prey_photons)-range_uv_prey,
                     np.array(uv_prey_photons)+range_uv_prey, color="r", alpha=0.2)
    plt.fill_between(distances, np.array(uv_scatter_photons)-range_uv_scatter,
                     np.array(uv_scatter_photons)+range_uv_scatter, color="b", alpha=0.2)
    plt.show()



max_distance = 1500
bkg_scatter = 0.00001
full_l = 1.0
normal_l = 0.5
dark_l = 0.38
uv_photon_scaling_factor = 50.0

plot_distinguishability_against_distance(max_distance, bkg_scatter, full_l, uv_photon_scaling_factor)
plot_distinguishability_against_distance(max_distance, bkg_scatter, normal_l, uv_photon_scaling_factor)
plot_distinguishability_against_distance(max_distance, bkg_scatter, dark_l, uv_photon_scaling_factor)


