import importlib

import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_assay_configuration_files

get_max_bkg_scatter = importlib.import_module(
    "Analysis.Calibration.Visual-Distance.full_model_bkg_count").get_max_bkg_scatter
full_model_prey_count = importlib.import_module(
    "Analysis.Calibration.Visual-Distance.full_model_prey_count").full_model_prey_count
compute_distinguishability = importlib.import_module(
    "Analysis.Calibration.Visual-Distance.visual_distance_calibration_first_principles_full_simplified").compute_distinguishability


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def compute_light_and_dark_gain(model_config, visual_distance_full, visual_distance_partial,
                                min_luminance, max_luminance, total_tests=1000):
    """For a given config, computes appropriate values for light and dark gain parameters.
    The light gain parameter will provide 100% distinguishability at visual_distance_full, and 75% distinguishability
    at visual_distance partial.
    The dark gain parameter will provide 75% distinguishability at visual_distance_full 50% distinugishability at
     visual_distance partial (impossible to distinguish prey from no prey).
    """
    luminance_vals = np.linspace(min_luminance, max_luminance, total_tests)

    learning_params, env_variables, n, b, c = load_assay_configuration_files(model_config)

    # env_variables['decay_rate'] = 0.008
    decay_constant = env_variables['decay_rate']
    env_variables["shot_noise"] = False
    env_variables["dark_light_ratio"] = 0.0
    env_variables["max_visual_distance"] = np.absolute(np.log(0.001) / env_variables["decay_rate"])

    bkg_scatter = env_variables["bkg_scatter"]

    uv_stimulus_photons_full = []
    uv_stimulus_photons_partial = []
    distinguishability_scores_full = []
    distinguishability_scores_partial = []

    # Baseline for zero distinguishability
    env_variables["bkg_scatter"] = bkg_scatter
    max_uv_scatter = get_max_bkg_scatter(env_variables["bkg_scatter"],
                                         decay_constant,
                                         env_variables["uv_photoreceptor_rf_size"],
                                         env_variables["width"],
                                         env_variables["height"],
                                         luminance_vals[0],
                                         env_variables)
    uv_scatter_photons = int(max_uv_scatter)
    baseline_distinguishability = compute_distinguishability(uv_scatter_photons, uv_scatter_photons)

    for l in luminance_vals:
        print(l)
        # No stimulus, but bkg_scatter present
        env_variables["bkg_scatter"] = bkg_scatter
        max_uv_scatter = get_max_bkg_scatter(env_variables["bkg_scatter"],
                                             decay_constant,
                                             env_variables["uv_photoreceptor_rf_size"],
                                             env_variables["width"],
                                             env_variables["height"],
                                             l,
                                             env_variables)
        uv_scatter_photons = int(max_uv_scatter)

        # Set bkg_scatter to zero
        env_variables["bkg_scatter"] = 0
        # Full visibility
        uv_prey = full_model_prey_count(0, decay_constant, env_variables["uv_photoreceptor_rf_size"],
                                        env_variables["width"],
                                        env_variables["height"], l,
                                        env_variables,
                                        visual_distance_full)
        uv_prey_full = int(uv_prey + uv_scatter_photons)

        # 50% visibility
        uv_prey_partial = full_model_prey_count(0, decay_constant, env_variables["uv_photoreceptor_rf_size"],
                                                env_variables["width"],
                                                env_variables["height"], l, env_variables,
                                                visual_distance_partial)
        uv_prey_partial_full = int(uv_prey_partial + uv_scatter_photons)
        print(f"Full: {uv_prey}, Partial: {uv_prey_partial}")

        uv_stimulus_photons_full.append(uv_prey_full)
        uv_stimulus_photons_partial.append(uv_prey_partial_full)

        distinguishability_scores_full.append(
            compute_distinguishability(uv_prey_full, uv_scatter_photons) - baseline_distinguishability)
        distinguishability_scores_partial.append(
            compute_distinguishability(uv_prey_partial_full, uv_scatter_photons) - baseline_distinguishability)

    plt.plot(luminance_vals, distinguishability_scores_full)
    plt.plot(luminance_vals, distinguishability_scores_partial)
    plt.title(f"Distinguishability")
    plt.legend([f"Distinguishability at {visual_distance_full / 10}mm",
                f"Distinguishability at {visual_distance_partial / 10}mm"])
    plt.xlabel("Luminance")
    plt.show()

    distinguishability_scores_partial = np.array(distinguishability_scores_partial) * 2
    distinguishability_scores_full = np.array(distinguishability_scores_full) * 2

    # Compute models to get values out.
    nearest_dark_i = find_nearest(distinguishability_scores_partial, 0.5)
    nearest_dark = luminance_vals[nearest_dark_i]
    print(f"{nearest_dark}-{distinguishability_scores_partial[nearest_dark_i]}")

    nearest_light_i = find_nearest(np.array(distinguishability_scores_full), 0.5)
    nearest_light = luminance_vals[nearest_light_i]
    print(f"{nearest_light}-{distinguishability_scores_full[nearest_light_i]}")


if __name__ == "__main__":
    config_name = "dqn_epsilon-1"
    visual_distance_full = 34
    visual_distance_partial = 100
    min_luminance = 0.01
    max_luminance = 12

    compute_light_and_dark_gain(config_name, visual_distance_full, visual_distance_partial, min_luminance,
                                max_luminance, total_tests=40)
