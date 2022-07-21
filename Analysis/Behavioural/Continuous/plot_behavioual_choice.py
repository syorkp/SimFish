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


def get_multiple_actions(p1, p2, p3, n=1):
    consumption_timestamps = np.array([])
    all_impulses = np.array([])
    all_angles = np.array([])
    predation_sequences = np.array([])
    predator_presence_timestamps = np.array([])
    for i in range(1, n+1):
        if i > 100:
            data = load_data(p1, f"{p2}-2", f"{p3} {i}")
        else:
            print(i)
            data = load_data(p1, p2, f"{p3}-{i}")
        all_impulses = np.concatenate((all_impulses, data["impulse"][1:]))
        all_angles = np.concatenate((all_angles, data["angle"][1:]))
        consumption_timestamps = np.concatenate((consumption_timestamps, [i for i, c in enumerate(data["consumed"]) if c == 1]))
        predator_presence_timestamps = np.concatenate((predator_presence_timestamps, [i for i, c in enumerate(data["predator_presence"]) if c == 1]))
        x = extract_consumption_action_sequences(data)
        x = [i for s in x for i in s]
        predation_sequences = np.concatenate((predation_sequences, x))

    return all_impulses, all_angles, consumption_timestamps, predation_sequences, predator_presence_timestamps


def get_multiple_means(p1, p2, p3, n=1):
    all_impulses = np.array([])
    all_angles = np.array([])
    for i in range(1, n+1):
        if i > 12:
            data = load_data(p1, f"{p2}-2", f"{p3} {i}")
        else:
            data = load_data(p1, p2, f"{p3}-{i}")
        all_impulses = np.concatenate((all_impulses, data["mu_impulse"][1:, 0]))
        all_angles = np.concatenate((all_angles, data["mu_angle"][1:, 0]))
    return all_impulses, all_angles


def extract_consumption_action_sequences(data, n=20):
    """Returns all action sequences that occur n steps before consumption"""
    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    prey_c_t = []
    action_sequences = []
    while len(consumption_timestamps) > 0:
        index = consumption_timestamps.pop(0)
        prey_capture_timestamps = [i for i in range(index-n+1, index+1) if i >= 0]
        prey_c_t.append(prey_capture_timestamps)
    return prey_c_t

# model_name = "ppo_continuous_sbe_is-1"
# model_name = "/scaffold_version_4-4"
model_name = "ppo_scaffold_18x-1"

data = load_data(model_name, "Behavioural-Data-Free", "Naturalistic-1")
# data = load_data("ppo_continuous_multivariate-7", "MultivariateData", "Naturalistic-1")
# data = load_data("ppo_multivariate_bptt-2", "MultivariateData", "Naturalistic-1")


# all_impulses, all_angles = get_multiple_actions("ppo_continuous_multivariate-7", "MultivariateData", "Naturalistic", 8)
# mu_impulse, mu_angle = get_multiple_means("ppo_continuous_multivariate-7", "MultivariateData", "Naturalistic", 8)

# all_impulses, all_angles, consumption_timestamps, predation_sequences = get_multiple_actions("ppo_continuous_multivariate-9", "MultivariateData", "Naturalistic", 8)
# mu_impulse, mu_angle = get_multiple_means("ppo_continuous_multivariate-9", "MultivariateData", "Naturalistic", 8)

all_impulses, all_angles, consumption_timestamps, predation_sequences, predator_presence = get_multiple_actions(model_name, "Behavioural-Data-Free", "Naturalistic", 12)
mu_impulse, mu_angle = get_multiple_means(model_name, "Behavioural-Data-Free", "Naturalistic", 12)

# all_impulses, all_angles, consumption_timestamps, predation_sequences, predator_presence = \
#     get_multiple_actions("ppo_continuous_beta_sanity-4", "Behavioural-Data-Free", "Naturalistic", 10)
# mu_impulse, mu_angle = \
#     get_multiple_means("ppo_continuous_beta_sanity-4", "Behavioural-Data-Free", "Naturalistic", 10)

consumption_timestamps = consumption_timestamps.astype(int)
consumption_timestamps = consumption_timestamps.tolist()

plt.scatter(all_impulses, all_angles, alpha=.2)
plt.show()


heatmap, xedges, yedges = np.histogram2d(mu_impulse*5, mu_angle, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()

mu_impulse = [i * 5 for i in mu_impulse]
mu_angle = [i * np.pi/5 for i in mu_angle]
plt.scatter(mu_impulse, mu_angle, alpha=.2)
consumption_mu_imp = [a for i, a in enumerate(mu_impulse) if i in consumption_timestamps]
consumption_mu_ang = [a for i, a in enumerate(mu_angle) if i in consumption_timestamps]
plt.scatter(consumption_mu_imp, consumption_mu_ang, alpha=.2, color="r")
predator_mu_imp = [a for i, a in enumerate(mu_impulse) if i in predator_presence]
predator_mu_ang = [a for i, a in enumerate(mu_angle) if i in predator_presence]
plt.scatter(predator_mu_imp, predator_mu_ang, alpha=.2, color="y")
plt.xlabel("Impulse")
plt.ylabel("Angle")
plt.show()


plt.scatter(mu_impulse, mu_angle, alpha=.2)
consumption_mu_imp = [a for i, a in enumerate(mu_impulse) if i in predation_sequences]
consumption_mu_ang = [a for i, a in enumerate(mu_angle) if i in predation_sequences]
plt.scatter(consumption_mu_imp, consumption_mu_ang, alpha=.2, color="r")
plt.show()


# sigma_impulse = data["sigma_impulse"]
# sigma_angle = data["sigma_angle"]
#
# plt.scatter(sigma_impulse, sigma_angle, alpha=.5)
# plt.show()

plot_all_consumption_sequences(all_impulses, all_angles, consumption_timestamps)

plt.hist(all_impulses, bins=60)
plt.show()

plt.hist(all_angles, bins=60)
plt.show()


plot_capture_sequences_orientation(data["fish_position"], all_angles, consumption_timestamps)
plot_capture_sequences_impulse(data["fish_position"], all_impulses, consumption_timestamps)
