import numpy as np

from Analysis.load_data import load_data

def get_in_light_vs_dark_steps(model_name, assay_config, assay_id, n):
    all_trials = []

    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_trials.append(label_in_light_steps(d))

    return all_trials

def label_in_light_steps(data):
    try:
        return data["in_light"].astype(int)
    except KeyError:
        return np.ones((data["observation"].shape[0])).astype(int)


def label_hemispheric_light_gradient(data, env_variables):
    observation = data["observation"]
    directional_brightness = np.mean(observation, axis=1)
    directional_brightness = directional_brightness[:, 2, 0] - directional_brightness[:, 2, 1]
    directional_brightness = (directional_brightness > 0) * 1
    return directional_brightness
