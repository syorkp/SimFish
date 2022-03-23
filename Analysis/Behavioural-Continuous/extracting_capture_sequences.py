import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Analysis.load_data import load_data


def extract_consumption_action_sequences(data, n=20):
    """Returns all action sequences that occur n steps before consumption"""
    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    prey_c_t = []

    impulse_sequences = []
    angle_sequences = []
    mu_impulse_sequences = []
    mu_angle_sequences = []
    sigma_impulse_sequences = []
    sigma_angle_sequences = []

    while len(consumption_timestamps) > 0:
        index = consumption_timestamps.pop(0)
        prey_capture_timestamps = [i for i in range(index-n+1, index+1) if i >= 0]
        prey_c_t.append(prey_capture_timestamps)

        impulse_sequence = [data["impulse"][i] for i in prey_capture_timestamps]
        impulse_sequences.append(impulse_sequence)

        angle_sequence = [data["angle"][i] for i in prey_capture_timestamps]
        angle_sequences.append(angle_sequence)

        mu_impulse_sequence = [data["mu_impulse"][i] for i in prey_capture_timestamps]
        mu_impulse_sequences.append(mu_impulse_sequence)

        mu_angle_sequence = [data["mu_angle"][i] for i in prey_capture_timestamps]
        mu_angle_sequences.append(mu_angle_sequence)

        sigma_impulse_sequence = [data["sigma_impulse"][i] for i in prey_capture_timestamps]
        sigma_impulse_sequences.append(sigma_impulse_sequence)

        sigma_angle_sequence = [data["sigma_angle"][i] for i in prey_capture_timestamps]
        sigma_angle_sequences.append(sigma_angle_sequence)

    return impulse_sequences, angle_sequences, prey_c_t, mu_impulse_sequences, mu_angle_sequences, sigma_impulse_sequences, sigma_angle_sequences



# data = load_data("ppo_continuous_multivariate-9", "MultivariateData", "Naturalistic-1")
#data = load_data("ppo_multivariate_bptt-2", "MultivariateData", "Naturalistic-1")
data = load_data("ppo_continuous_multivariate-7", "MultivariateData", "Naturalistic-1")

im, an, pre, mu_impulse_sequences, mu_angle_sequences, sigma_impulse_sequences, sigma_angle_sequences = extract_consumption_action_sequences(data, 10)

im = [i for s in im for i in s]
an = [i for s in an for i in s]
plt.scatter(im, an, alpha=.5)
plt.show()

mu_impulse = data["mu_impulse"]
mu_angle = data["mu_angle"]

plt.scatter(mu_impulse, mu_angle, alpha=.5)
mu_impulse_sequences = [i for s in mu_impulse_sequences for i in s]
mu_angle_sequences = [i for s in mu_angle_sequences for i in s]
plt.scatter(mu_impulse_sequences, mu_angle_sequences, alpha=.5, color="r")
plt.show()

sigma_impulse_sequences = [i for s in sigma_impulse_sequences for i in s]
sigma_angle_sequences = [i for s in sigma_angle_sequences for i in s]
plt.scatter(sigma_impulse_sequences, sigma_angle_sequences, alpha=.5)
plt.show()
