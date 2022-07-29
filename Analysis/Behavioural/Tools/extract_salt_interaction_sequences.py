

def label_salt_health_decreasing(data, env_variables):
    salt_concentration = data["salt"]
    salt_decreasing = (salt_concentration > env_variables["salt_recovery"])
    return salt_decreasing



