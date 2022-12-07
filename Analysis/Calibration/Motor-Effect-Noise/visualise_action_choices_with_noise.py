import numpy as np
import matplotlib.pyplot as plt


def get_normal_mean_sample(value):
    samples = np.random.normal(0, value, 100)
    samples = np.absolute(samples)
    return np.mean(samples)


def apply_motor_effect_noise(impulse_effect_noise_i, impulse_effect_noise_a, angle_effect_noise_a, angle_effect_noise_i):
    impulse_effect_noise_i_mean = get_normal_mean_sample(impulse_effect_noise_i)
    impulse_effect_noise_a_mean = get_normal_mean_sample(impulse_effect_noise_a)
    angle_effect_noise_a_mean = get_normal_mean_sample(angle_effect_noise_a)
    angle_effect_noise_i_mean = get_normal_mean_sample(angle_effect_noise_i)

    i = np.linspace(0, 100, 100)
    a = np.linspace(0, 2, 100)

    # Form grid
    impulses, angles = np.meshgrid(i, a)

    # Create two maps from grid, which show measure of deviation in each parameter, given the indicated pair
    mean_deviation_impulse = (impulses * impulse_effect_noise_i_mean) + (angles * impulse_effect_noise_a_mean)
    mean_deviation_angle = (angles * angle_effect_noise_a_mean) + (impulses * angle_effect_noise_i_mean)


    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(mean_deviation_impulse)
    plt.xlabel("Impulse")
    plt.ylabel("Angle")
    plt.xticks(range(0, 100, 10), [round(y, 2) for x, y in enumerate(i) if x % 10 == 0])
    plt.yticks(range(0, 100, 10), [round(y, 2) for x, y in enumerate(a) if x % 10 == 0])
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(mean_deviation_angle)
    plt.xlabel("Impulse")
    plt.ylabel("Angle")
    plt.xticks(range(0, 100, 10), [round(y, 2) for x, y in enumerate(i) if x % 10 == 0])
    plt.yticks(range(0, 100, 10), [round(y, 2) for x, y in enumerate(a) if x % 10 == 0])
    plt.colorbar()
    plt.show()
    x = True


if __name__ == "__main__":
    apply_motor_effect_noise(0.1, 0.5, 0.6, 0.02)
