import numpy as np
import matplotlib.pyplot as plt


def reset_salt_gradient(max_salt_damage, salt_recovery_rate, salt_concentration_decay, width=3000, height=3000):


    xp, yp = np.arange(width), np.arange(height)

    salt_source_x = np.random.randint(0, width - 1)
    salt_source_y = np.random.randint(0, height - 1)
    salt_location = [salt_source_x, salt_source_y]
    salt_distance = (((salt_source_x - xp[:, None]) ** 2 + (
            salt_source_y - yp[None, :]) ** 2) ** 0.5)  # Measure of distance from source at every point.
    salt_gradient = np.exp(-salt_concentration_decay * salt_distance) * max_salt_damage

    fig, axs = plt.subplots(2, figsize=(10, 20))
    axs[0].set_title("Salt Distribution")
    axs[0].imshow(salt_gradient)

    salt_unsafe_locations = (salt_gradient > salt_recovery_rate) * 1
    axs[1].set_title("Salt Unsafe Locations")
    axs[1].imshow(salt_unsafe_locations)

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.show()


if __name__ == "__main__":
    reset_salt_gradient(max_salt_damage=0.02, salt_recovery_rate=0.005, salt_concentration_decay=0.004)
