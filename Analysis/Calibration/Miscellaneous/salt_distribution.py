import numpy as np


def reset_salt_gradient():
    salt_concentration_decay = 0.005
    max_salt_damage = 0.02
    xp, yp = np.arange(1500), np.arange(1500)

    salt_source_x = np.random.randint(0, 1500 - 1)
    salt_source_y = np.random.randint(0, 1500 - 1)
    salt_location = [salt_source_x, salt_source_y]
    salt_distance = (((salt_source_x - xp[:, None]) ** 2 + (
            salt_source_y - yp[None, :]) ** 2) ** 0.5)  # Measure of distance from source at every point.
    salt_gradient = np.exp(-salt_concentration_decay * salt_distance) * \
                         max_salt_damage

    x = True

reset_salt_gradient()