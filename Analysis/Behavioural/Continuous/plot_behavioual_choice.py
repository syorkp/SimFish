import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Analysis.load_data import load_data


def plot_capture_sequences_orientation(position, orientation_changes, consumption_timestamps):
    # Note that due to the inverse scale in the environment, should be rotated in y axis.
    data = {}
    # sns.set()
    data["x"] = []
    data["y"] = []
    data["Delta-Angle"] = []
    for c in consumption_timestamps:
        data["x"] += [p[0] for i, p in enumerate(position) if i in range(c-15, c)]
        data["y"] += [p[1] for i, p in enumerate(position) if i in range(c-15, c)]
        data["Delta-Angle"] += [o for i, o in enumerate(orientation_changes) if i in range(c-15, c)]

    consumption_positions = [p for i, p in enumerate(position) if i in consumption_timestamps]

    plt.figure(figsize=(10, 10))
    sns.relplot(x="x", y="y", size="Delta-Angle",
                sizes=(40, 400), alpha=.5, palette="muted",
                height=6, data=data)
    plt.scatter([p[0] for p in consumption_positions], [p[1] for p in consumption_positions], color="r")
    plt.show()

def plot_capture_sequences_impulse(position, impulses, consumption_timestamps):
    # Note that due to the inverse scale in the environment, should be rotated in y axis.
    data = {}
    # sns.set()
    data["x"] = []
    data["y"] = []
    data["Impulses"] = []
    for c in consumption_timestamps:
        data["x"] += [p[0] for i, p in enumerate(position) if i in range(c-15, c)]
        data["y"] += [p[1] for i, p in enumerate(position) if i in range(c-15, c)]
        data["Impulses"] += [o for i, o in enumerate(impulses) if i in range(c-15, c)]

    consumption_positions = [p for i, p in enumerate(position) if i in consumption_timestamps]

    plt.figure(figsize=(10, 10))
    sns.relplot(x="x", y="y", size="Impulses",
                sizes=(40, 400), alpha=.5, palette="muted",
                height=6, data=data)
    plt.scatter([p[0] for p in consumption_positions], [p[1] for p in consumption_positions], color="r")
    plt.show()


def plot_all_consumption_sequences(all_impulses, all_angles, consumption_times):

    for c in consumption_times:
        impulse_sequence = [imp for i, imp in enumerate(all_impulses) if i in range(c-20, c)]
        plt.plot(impulse_sequence)

    plt.show()
    for c in consumption_times:
        angle_sequence = [ang for i, ang in enumerate(all_angles) if i in range(c-20, c)]
        plt.plot(angle_sequence)
    plt.show()


# data = load_data("ppo_continuous_multivariate-9", "MultivariateData", "Naturalistic-1")
data = load_data("ppo_continuous_multivariate-7", "MultivariateData", "Naturalistic-1")
# data = load_data("ppo_multivariate_bptt-2", "MultivariateData", "Naturalistic-1")

all_impulses = data["impulse"]
all_angles = data["angle"]
consumption_timestamps = [i for i, c in enumerate(data["consumed"]) if c == 1]

plt.scatter(all_impulses, all_angles, alpha=.5)
plt.show()

mu_impulse = data["mu_impulse"]
mu_angle = data["mu_angle"]

plt.scatter(mu_impulse, mu_angle, alpha=.5)
plt.show()

sigma_impulse = data["sigma_impulse"]
sigma_angle = data["sigma_angle"]

plt.scatter(sigma_impulse, sigma_angle, alpha=.5)
plt.show()

plot_all_consumption_sequences(all_impulses, all_angles, consumption_timestamps)

plt.hist(all_impulses, bins=20)
plt.show()

plt.hist(all_angles, bins=20)
plt.show()


plot_capture_sequences_orientation(data["fish_position"], all_angles, consumption_timestamps)
plot_capture_sequences_impulse(data["fish_position"], all_impulses, consumption_timestamps)
