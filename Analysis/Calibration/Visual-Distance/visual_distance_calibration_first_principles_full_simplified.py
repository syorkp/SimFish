"""
Simpler version of other, which gets actual max bgk_scatter. Note that it ignores occlusion entirely.
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

from Analysis.load_model_config import load_configuration_files
get_max_bkg_scatter = importlib.import_module("Analysis.Calibration.Visual-Distance.full_model_bkg_count").get_max_bkg_scatter


def prey_signal(L, d, decay_constant):
    return L * np.exp(-decay_constant * d)


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


def plot_distinguishability_against_distance(max_distance, bkg_scatter, luminance, scaling_factor,
                                             rf_size, decay_constant, max_curve_distance=1000):
    prey_size = 1

    distinguishability_scores = []
    distances = np.linspace(10, max_curve_distance, 10)

    uv_prey_photons = []
    uv_scatter_photons = []

    learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_10-1")

    max_uv_scatter = get_max_bkg_scatter(bkg_scatter, decay_constant, rf_size, 1500, 1500, luminance, env_variables)
    max_uv_scatter *= scaling_factor

    for distance in distances:
        uv_prey = prey_signal(luminance, distance, decay_constant)
        uv_prey *= scaling_factor

        max_uv_scatter = int(max_uv_scatter)
        uv_prey_stimulus = int(uv_prey + max_uv_scatter)

        distinguishability_scores.append(compute_distinguishability(uv_prey_stimulus, max_uv_scatter))

        uv_prey_photons.append(uv_prey_stimulus)
        uv_scatter_photons.append(max_uv_scatter)

    plt.plot(distances, distinguishability_scores)
    plt.title(f"BKG: {bkg_scatter}, Luminance: {luminance}")
    plt.xlabel("Distance (mm x10^1)")
    plt.ylabel("Percentage Easily Discriminable")
    plt.show()

    # Absolute photon values
    range_uv_prey = 1.96 * np.sqrt(np.abs(np.array(uv_prey_photons)))
    range_uv_scatter = 1.96 * np.sqrt(np.abs(np.array(uv_scatter_photons)))
    plt.plot(distances, uv_prey_photons, color="r")
    plt.plot(distances, uv_scatter_photons, color="b")
    plt.title(f"Absolute photon values. BKG: {bkg_scatter}, Luminance: {luminance}")
    # plt.vlines(visual_distance, min(distinguishability_scores), max(uv_prey_photons) + max(range_uv_prey), color="green")
    plt.fill_between(distances, np.array(uv_prey_photons)-range_uv_prey,
                     np.array(uv_prey_photons)+range_uv_prey, color="r", alpha=0.2)
    plt.fill_between(distances, np.array(uv_scatter_photons)-range_uv_scatter,
                     np.array(uv_scatter_photons)+range_uv_scatter, color="b", alpha=0.2)
    plt.xlabel("Distance (mm x10^1)")
    plt.ylabel("Photons per stimulus")
    plt.show()


def plot_distinguishability_against_luminance(visual_distance, max_distance, bkg_scatter, scaling_factor,
                                              rf_size, decay_constant, min_luminance, max_luminance):
    prey_size = 1

    luminance_vals = np.linspace(min_luminance, max_luminance, 100)

    uv_stimulus_photons = []
    distinguishability_scores = []
    learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_10-1")

    for l in luminance_vals:
        uv_prey = prey_signal(l, visual_distance, decay_constant)
        uv_prey *= scaling_factor

        max_uv_scatter = get_max_bkg_scatter(bkg_scatter, decay_constant, rf_size, 1500, 1500, l, env_variables)
        max_uv_scatter *= scaling_factor
        max_uv_scatter = int(max_uv_scatter)

        uv_prey_stimulus = int(uv_prey + max_uv_scatter)
        uv_stimulus_photons.append(uv_prey_stimulus)
        distinguishability_scores.append(compute_distinguishability(uv_prey_stimulus, max_uv_scatter))

    plt.plot(luminance_vals, distinguishability_scores)
    plt.title(f"Distinguishability at {visual_distance/10}mm")
    plt.xlabel("Luminance")
    plt.show()

    plt.plot(luminance_vals, [max_uv_scatter for i in range(len(luminance_vals))])
    plt.plot(luminance_vals, uv_stimulus_photons, color="r")
    plt.legend(["Max Scatter Photons", "Max Prey Photons"])
    plt.title(f"Photon counts at {visual_distance/10}mm")
    plt.show()


def plot_distinguishability_against_luminance_two_distances(visual_distance_full, visual_distance_partial, max_distance,
                                                            bkg_scatter, scaling_factor, uv_occlusion_gain, min_luminance,
                                                            max_luminance, rf_size, decay_constant):
    prey_size = 1
    learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_10-1")

    luminance_vals = np.linspace(min_luminance, max_luminance, 100)
    uv_scatter_photons =  get_max_bkg_scatter(bkg_scatter, decay_constant, rf_size, 1500, 1500, min_luminance, env_variables)
    uv_scatter_photons = int(uv_scatter_photons)
    uv_stimulus_photons_full = []
    uv_stimulus_photons_partial = []
    distinguishability_scores_full = []
    distinguishability_scores_partial = []

    for l in luminance_vals:
        # Full visibility
        uv_prey = prey_signal(l, visual_distance_full, decay_constant)
        uv_prey *= scaling_factor
        uv_prey_full = int(uv_prey+ uv_scatter_photons)

        max_uv_scatter = get_max_bkg_scatter(bkg_scatter, decay_constant, rf_size, 1500, 1500, l, env_variables)
        max_uv_scatter *= scaling_factor
        max_uv_scatter_full = int(max_uv_scatter)


        # 50% visibility
        uv_prey = prey_signal(l, visual_distance_full, decay_constant)
        uv_prey *= scaling_factor
        uv_prey_partial = int(uv_prey)

        max_uv_scatter = get_max_bkg_scatter(bkg_scatter, decay_constant, rf_size, 1500, 1500, l, env_variables)
        max_uv_scatter *= scaling_factor
        max_uv_scatter_partial = int(max_uv_scatter)

        uv_stimulus_photons_full.append(uv_prey_full)
        uv_stimulus_photons_partial.append(uv_prey_partial)

        distinguishability_scores_full.append(compute_distinguishability(uv_prey_full, uv_scatter_photons))
        distinguishability_scores_partial.append(compute_distinguishability(uv_prey_partial, uv_scatter_photons))

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


L1 = 200
max_distance = (1500**2 + 1500**2) ** 0.5
bkg_scatter = 0.1
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

plot_distinguishability_against_distance(max_distance, bkg_scatter, L1, scaling_factor, rf_size, decay)
plot_distinguishability_against_luminance_two_distances(visual_distance_full, visual_distance_partial, max_distance,
                                                        bkg_scatter, scaling_factor, uv_occlusion_gain, min_luminance,
                                                        max_luminance, rf_size, decay)