import numpy as np


def label_in_light_steps(data):
    try:
        return data["in_light"]
    except KeyError:
        return np.ones((data["observation"].shape[0]))


def label_hemispheric_light_gradient(data, env_variables):
    observation = data["observation"]
    directional_brightness = np.mean(observation, axis=1)
    directional_brightness = directional_brightness[:, 2, 0] - directional_brightness[:, 2, 1]
    directional_brightness = (directional_brightness > 0) * 1
    return directional_brightness
