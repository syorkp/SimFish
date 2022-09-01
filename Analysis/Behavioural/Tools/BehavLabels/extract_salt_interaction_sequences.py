

def label_salt_health_decreasing(data, env_variables):
    salt_concentration = data["salt"]
    salt_decreasing = (salt_concentration > env_variables["salt_recovery"])
    return salt_decreasing


def label_salt_health_within_range(data, s_min, s_max):
    salt_health = data["salt_health"]
    below_max = salt_health < s_max
    above_min = salt_health >= s_min
    within_range = (below_max * above_min) * 1
    return within_range

