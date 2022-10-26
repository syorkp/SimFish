import numpy as np

import matplotlib.pyplot as plt



def plot_motor_effect_noise_scatter(impulse_std, angle_std):
    # set_impulse = np.linspace(0, 50, 100)
    set_angle = np.linspace(0, 1, 10)

    fig, axs = plt.subplots(1, 2)

    for i in range(5000):
        # impulse = set_impulse + (np.random.normal(0, impulse_std) * np.absolute(set_impulse))
        angle = set_angle + (np.random.normal(0, angle_std) * np.absolute(set_angle))

        # axs[0].scatter(set_impulse, impulse, color="b", alpha=0.2)
        axs[1].scatter(set_angle, angle, color="b", alpha=0.02)
    plt.show()


if __name__ == "__main__":
    plot_motor_effect_noise_scatter(0.14, 0.5)
