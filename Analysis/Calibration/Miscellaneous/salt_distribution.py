import numpy as np
import matplotlib.pyplot as plt


def reset_salt_gradient():
    salt_concentration_decay = 0.002
    max_salt_damage = 0.02
    salt_recovery_rate = 0.01
    xp, yp = np.arange(1500), np.arange(1500)

    salt_source_x = np.random.randint(0, 1500 - 1)
    salt_source_y = np.random.randint(0, 1500 - 1)
    salt_location = [salt_source_x, salt_source_y]
    salt_distance = (((salt_source_x - xp[:, None]) ** 2 + (
            salt_source_y - yp[None, :]) ** 2) ** 0.5)  # Measure of distance from source at every point.
    salt_gradient = np.exp(-salt_concentration_decay * salt_distance) * max_salt_damage

    plt.title("Salt Distribution")
    plt.imshow(salt_gradient)
    plt.show()

    salt_unsafe_locations = (salt_gradient > salt_recovery_rate) * 1
    plt.title("Salt Unsafe Locations")
    plt.imshow(salt_unsafe_locations)
    plt.show()
    x = True

reset_salt_gradient()
