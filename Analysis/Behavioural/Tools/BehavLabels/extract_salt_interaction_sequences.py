import numpy as np

def label_salt_health_decreasing(data, env_variables):
    try:
        salt_concentration = data["salt"]
        salt_decreasing = (salt_concentration > env_variables["salt_recovery"])
    except KeyError:
        salt_decreasing = np.zeros((data["observation"].shape[0]))
    return salt_decreasing


def label_salt_health_within_range(data, s_min, s_max):
    try:
        salt_health = data["salt_health"]
        below_max = salt_health < s_max
        above_min = salt_health >= s_min
        within_range = (below_max * above_min) * 1
    except KeyError:
        within_range = np.zeros((data["observation"].shape[0]))

    return within_range

