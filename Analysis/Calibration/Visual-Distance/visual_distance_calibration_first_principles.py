"""First model attempt to get discriminability from first principles. """

import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

from Analysis.load_model_config import load_assay_configuration_files
get_max_background_brightness = importlib.import_module("Analysis.Calibration.Visual-Distance.full_model_bkg_count").get_max_background_brightness


def scatter_signal_all(max_d, rf_size, background_brightness):
    d_range = np.linspace(1, max_d, max_d-1)

    point_width = 2 * d_range * np.tan(rf_size / 2)
    distance_scaling = np.exp(-0.0006 * d_range) * background_brightness
    point_width = np.clip(point_width, 1, 10000)
    point_width += (point_width > 1) * 2
    point_width = np.floor(point_width).astype(int)
    photons = np.sum(distance_scaling * point_width)

    return photons


def get_max_scatter_photons(background_brightness, distance, rf_size, decay_constant):
    photons = 0

    for d in range(int(distance)):
        point_width = 2 * d * np.tan(rf_size/2)
        distance_scaling = np.exp(-decay_constant * d) * background_brightness
        if point_width < 1:
            point_width = 1
        else:
            point_width = int(point_width) + 2
        photons += distance_scaling * point_width

    return photons


def get_max_scatter_photons_occluded(background_brightness, rf_size, prey_distance, prey_radius, max_distance, decay_constant, uv_occlusion_gain):
    photons = 0

    prey_angular_size = 2 * np.arctan((prey_radius/2)/prey_distance)
    fraction_occluded = prey_angular_size/rf_size
    if fraction_occluded > 1.0:
        fraction_occluded = 1.0
    fraction_visible = 1-fraction_occluded

    # if fraction_occluded > 1:
    #     return 0

    for d in range(int(prey_distance), int(max_distance)):
        point_width = 2 * d * np.tan(rf_size/2)
        distance_scaling = np.exp(-decay_constant * d) * background_brightness
        if point_width < 1:
            point_width = 1
        photons += fraction_visible * distance_scaling * point_width
        photons += fraction_occluded * distance_scaling * point_width * uv_occlusion_gain

    return photons


def get_max_prey_photons(distance, rf_size, decay_constant, luminance):

    # width_at_prey = 2 * distance * np.tan(rf_size/2)

    pixel_value = np.exp(-decay_constant * distance) * luminance

    return pixel_value


def compute_distinguishability(prey_stimulus, max_noise_stimulus):
    distinguishability = 0
    max_pixels = int(max(prey_stimulus, max_noise_stimulus) + 1.96 * np.sqrt(max(prey_stimulus, max_noise_stimulus)))
    min_pixels = int(min(prey_stimulus, max_noise_stimulus) - 1.96 * np.sqrt(min(prey_stimulus, max_noise_stimulus)))
    for p in range(min_pixels, max_pixels):
        pa = poisson.pmf(p, prey_stimulus)
        pb = poisson.pmf(p, max_noise_stimulus)
        if pa == 0:
            pa += 0.000000001
        if pb == 0:
            pb += 0.000000001
        distinguishability += pa * (pa/(pa + pb))
    return distinguishability


def compute_distinguishability_old(prey_stimulus, max_noise_stimulus):
    distinguishability = (prey_stimulus - max_noise_stimulus)
    return distinguishability


def plot_distinguishability_against_distance(max_distance, background_brightness, luminance, scaling_factor, uv_occlusion_gain, rf_size, decay_constant, max_curve_distance=1000):
    prey_radius = 1

    distinguishability_scores = []
    distances = np.linspace(10, max_curve_distance, 10)

    uv_prey_photons = []
    uv_scatter_photons = []

    for distance in distances:
        max_uv_scatter = get_max_scatter_photons(background_brightness, max_distance, rf_size, decay_constant)
        uv_scatter_before_prey = get_max_scatter_photons(background_brightness, distance, rf_size, decay_constant)
        uv_scatter_after_prey = get_max_scatter_photons_occluded(background_brightness, rf_size, distance, prey_radius, max_distance, decay_constant, uv_occlusion_gain)
        uv_prey = get_max_prey_photons(distance, rf_size, decay_constant, luminance)
        uv_prey_stimulus = uv_prey + uv_scatter_after_prey + uv_scatter_before_prey

        max_uv_scatter *= scaling_factor
        uv_prey_stimulus *= scaling_factor
        max_uv_scatter = int(max_uv_scatter)
        uv_prey_stimulus = int(uv_prey_stimulus)

        distinguishability_scores.append(compute_distinguishability(uv_prey_stimulus, max_uv_scatter))

        uv_prey_photons.append(uv_prey_stimulus)
        uv_scatter_photons.append(max_uv_scatter)

    plt.plot(distances, distinguishability_scores)
    plt.title(f"BKG: {background_brightness}, Luminance: {luminance}")
    plt.xlabel("Distance (mm x10^1)")
    plt.ylabel("Percentage Easily Discriminable")
    plt.show()

    # Absolute photon values
    range_uv_prey = 1.96 * np.sqrt(np.abs(np.array(uv_prey_photons)))
    range_uv_scatter = 1.96 * np.sqrt(np.abs(np.array(uv_scatter_photons)))
    plt.plot(distances, uv_prey_photons, color="r")
    plt.plot(distances, uv_scatter_photons, color="b")
    plt.title(f"Absolute photon values. BKG: {background_brightness}, Luminance: {luminance}")
    # plt.vlines(visual_distance, min(distinguishability_scores), max(uv_prey_photons) + max(range_uv_prey), color="green")
    plt.fill_between(distances, np.array(uv_prey_photons)-range_uv_prey,
                     np.array(uv_prey_photons)+range_uv_prey, color="r", alpha=0.2)
    plt.fill_between(distances, np.array(uv_scatter_photons)-range_uv_scatter,
                     np.array(uv_scatter_photons)+range_uv_scatter, color="b", alpha=0.2)
    plt.xlabel("Distance (mm x10^1)")
    plt.ylabel("Photons per stimulus")
    plt.show()


def plot_distinguishability_against_luminance(visual_distance, max_distance, background_brightness, scaling_factor, uv_occlusion_gain, min_luminance, max_luminance):
    rf_size = 0.0128
    decay_constant = 0.0006
    prey_radius = 1

    luminance_vals = np.linspace(min_luminance, max_luminance, 100)
    uv_scatter_photons = get_max_scatter_photons(background_brightness, max_distance, rf_size, decay_constant) * scaling_factor
    uv_scatter_photons = int(uv_scatter_photons)
    uv_stimulus_photons = []
    distinguishability_scores = []

    for l in luminance_vals:
        uv_scatter_before_prey = get_max_scatter_photons(background_brightness, visual_distance, rf_size, decay_constant)
        uv_scatter_after_prey = get_max_scatter_photons_occluded(background_brightness, rf_size, visual_distance, prey_radius, max_distance, decay_constant, uv_occlusion_gain)
        uv_prey = get_max_prey_photons(visual_distance, rf_size, decay_constant, l)
        uv_prey_stimulus = uv_prey + uv_scatter_after_prey + uv_scatter_before_prey

        uv_prey_stimulus *= scaling_factor
        uv_prey_stimulus = int(uv_prey_stimulus)
        uv_stimulus_photons.append(uv_prey_stimulus)
        distinguishability_scores.append(compute_distinguishability(uv_prey_stimulus, uv_scatter_photons))

    plt.plot(luminance_vals, distinguishability_scores)
    plt.title(f"Distinguishability at {visual_distance/10}mm")
    plt.xlabel("Luminance")
    plt.show()

    plt.plot(luminance_vals, [uv_scatter_photons for i in range(len(luminance_vals))])
    plt.plot(luminance_vals, uv_stimulus_photons, color="r")
    plt.legend(["Max Scatter Photons", "Max Prey Photons"])
    plt.title(f"Photon counts at {visual_distance/10}mm")
    plt.show()


def plot_distinguishability_against_luminance_two_distances(visual_distance_full, visual_distance_partial, max_distance,
                                                            background_brightness, scaling_factor, uv_occlusion_gain, min_luminance,
                                                            max_luminance):
    rf_size = 0.0128
    decay_constant = 0.0006
    prey_radius = 1

    luminance_vals = np.linspace(min_luminance, max_luminance, 100)
    uv_scatter_photons = get_max_scatter_photons(background_brightness, max_distance, rf_size, decay_constant) * scaling_factor
    uv_scatter_photons = int(uv_scatter_photons)
    uv_stimulus_photons_full = []
    uv_stimulus_photons_partial = []
    distinguishability_scores_full = []
    distinguishability_scores_partial = []

    for l in luminance_vals:
        # Full visibility
        uv_scatter_before_prey = get_max_scatter_photons(background_brightness, visual_distance_full, rf_size, decay_constant)
        uv_scatter_after_prey = get_max_scatter_photons_occluded(background_brightness, rf_size, visual_distance_full, prey_radius, max_distance, decay_constant, uv_occlusion_gain)
        uv_prey = get_max_prey_photons(visual_distance_full, rf_size, decay_constant, l)
        uv_prey_stimulus_full = uv_prey + uv_scatter_after_prey + uv_scatter_before_prey

        # 50% visibility
        uv_scatter_before_prey = get_max_scatter_photons(background_brightness, visual_distance_partial, rf_size, decay_constant)
        uv_scatter_after_prey = get_max_scatter_photons_occluded(background_brightness, rf_size, visual_distance_partial, prey_radius, max_distance, decay_constant, uv_occlusion_gain)
        uv_prey = get_max_prey_photons(visual_distance_partial, rf_size, decay_constant, l)
        uv_prey_stimulus_partial = uv_prey + uv_scatter_after_prey + uv_scatter_before_prey

        uv_prey_stimulus_full *= scaling_factor
        uv_prey_stimulus_full = int(uv_prey_stimulus_full)

        uv_prey_stimulus_partial *= scaling_factor
        uv_prey_stimulus_partial = int(uv_prey_stimulus_partial)

        uv_stimulus_photons_full.append(uv_prey_stimulus_full)
        uv_stimulus_photons_partial.append(uv_prey_stimulus_partial)

        distinguishability_scores_full.append(compute_distinguishability(uv_prey_stimulus_full, uv_scatter_photons))
        distinguishability_scores_partial.append(compute_distinguishability(uv_prey_stimulus_partial, uv_scatter_photons))

    plt.plot(luminance_vals, distinguishability_scores_full)
    plt.plot(luminance_vals, distinguishability_scores_partial)
    plt.title(f"Distinguishability")
    plt.legend([f"Distinguishability at {visual_distance_full/10}mm", f"Distinguishability at {visual_distance_partial/10}mm"])
    plt.xlabel("Luminance")
    plt.show()

    plt.plot(luminance_vals, [uv_scatter_photons for i in range(len(luminance_vals))])
    plt.plot(luminance_vals, uv_stimulus_photons_full, color="r")
    plt.plot(luminance_vals, uv_stimulus_photons_partial, color="g")
    plt.legend(["Max Scatter Photons", f"Max Prey Photons {visual_distance_full/10}mm", f"Max Prey Photons {visual_distance_partial/10}mm"])
    plt.title(f"Photon counts")
    plt.show()

    with open(
            f"Data-Various/distinguishability_scores_partial.npy",
            "wb") as f:
        np.save(f, np.array(distinguishability_scores_partial))

    with open(
            f"Data-Various/distinguishability_scores_full.npy",
            "wb") as f:
        np.save(f, np.array(distinguishability_scores_full))

    with open(
            f"Data-Various/uv_stimulus_photons_full.npy",
            "wb") as f:
        np.save(f, np.array(uv_stimulus_photons_full))

    with open(
            f"Data-Various/uv_stimulus_photons_partial.npy",
            "wb") as f:
        np.save(f, np.array(uv_stimulus_photons_partial))


#                 OLD

# max_distance = 1500
# background_brightness = 0.00019
# full_l = 1.0
# normal_l = 0.27212121212121215
# dark_l = 0.2693939393939394
# scaling_factor = 1000000
# uv_occlusion_gain = 1.0
# visual_distance_full = 34
# visual_distance_partial = 100
#
# min_luminance = 0.25
# max_luminance = 0.2721
#
#
# plot_distinguishability_against_distance(max_distance, background_brightness, full_l, scaling_factor, uv_occlusion_gain)
# plot_distinguishability_against_distance(max_distance, background_brightness, max_luminance, scaling_factor, uv_occlusion_gain)
#
# plot_distinguishability_against_luminance_two_distances(visual_distance_full, visual_distance_partial, max_distance, background_brightness, scaling_factor, uv_occlusion_gain, min_luminance, max_luminance)
#
#
# # plot_distinguishability_against_luminance(visual_distance_full, max_distance, background_brightness, scaling_factor, uv_occlusion_gain, min_luminance, max_luminance)
# # plot_distinguishability_against_luminance(visual_distance_partial, max_distance, background_brightness, scaling_factor, uv_occlusion_gain, min_luminance, max_luminance)
# #
# plot_distinguishability_against_distance(max_distance, background_brightness, full_l, scaling_factor, uv_occlusion_gain)
# plot_distinguishability_against_distance(max_distance, background_brightness, normal_l, scaling_factor, uv_occlusion_gain)
# plot_distinguishability_against_distance(max_distance, background_brightness, dark_l, scaling_factor, uv_occlusion_gain)

L1 = 200
max_distance = (1500**2 + 1500**2) ** 0.5
background_brightness = 0.1
scaling_factor = 1
uv_occlusion_gain = 1.0
visual_distance_full = 34
visual_distance_partial = 100
min_luminance = 10
max_luminance = 200
decay = 0.01
luminance = 200
distance = 600
rf_size = 0.0133 * 3

plot_distinguishability_against_distance(max_distance, background_brightness, L1, scaling_factor, uv_occlusion_gain, rf_size, decay)
plot_distinguishability_against_luminance_two_distances(visual_distance_full, visual_distance_partial, max_distance, background_brightness, scaling_factor, uv_occlusion_gain, min_luminance, max_luminance)

