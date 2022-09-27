import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.label_behavioural_context import label_behavioural_context_multiple_trials, \
    get_behavioural_context_name_by_index

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
            data = load_data(p1, p2, f"{p3}-{i}")
        all_impulses = np.concatenate((all_impulses, data["impulse"][1:]))
        all_angles = np.concatenate((all_angles, data["angle"][1:]))
        consumption_timestamps = np.concatenate((consumption_timestamps, [i for i, c in enumerate(data["consumed"]) if c == 1]))
        predator_presence_timestamps = np.concatenate((predator_presence_timestamps, [i for i, c in enumerate(data["predator_presence"]) if c == 1]))
        x = extract_consumption_action_sequences(data)
        x = [i for s in x for i in s]
        predation_sequences = np.concatenate((predation_sequences, x))

    consumption_timestamps = consumption_timestamps.astype(int)
    consumption_timestamps = consumption_timestamps.tolist()

    return all_impulses, all_angles, consumption_timestamps, predation_sequences, predator_presence_timestamps


def get_multiple_means(p1, p2, p3, n=1):
    all_impulses = np.array([])
    all_angles = np.array([])
    for i in range(1, n+1):
        if i > 100:
            data = load_data(p1, f"{p2}-2", f"{p3} {i}")
        else:
            data = load_data(p1, p2, f"{p3}-{i}")
        all_impulses = np.concatenate((all_impulses, data["mu_impulse"][:, 0]))
        all_angles = np.concatenate((all_angles, data["mu_angle"][:, 0]))
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


def plot_action_scatter(impulses, angles, model_name, special_impulses=None, special_angles=None, special_names=None):
    plot_name = f"{model_name}-action_scatter"

    plt.scatter(impulses, angles, alpha=.1)

    special_colours = ["r", "g", "y"]
    i = 0
    if special_names is not None:
        for impulses_s, angles_s in zip(special_impulses, special_angles):
            plt.scatter(impulses_s, angles_s, color=special_colours[i], alpha=0.2)
            i += 1
        plot_name = "-".join(special_names) + "-" + plot_name

    plt.xlabel("Impulse")
    plt.ylabel("Angle (pi radians)")
    plt.savefig(f"All-Plots/{model_name}/{plot_name}.jpg")
    plt.clf()


def plot_action_use_density(mu_impulse, mu_angle, model_name, n_bins=100):
    heatmap, xedges, yedges = np.histogram2d(mu_impulse * max_impulse, mu_angle, bins=n_bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    hmp = np.clip(heatmap.T, 0, 100)
    hmp = heatmap.T

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(hmp, extent=extent, origin='lower', aspect=8)
    plt.xlabel("Impulse")
    plt.ylabel("Angle (pi radians)")
    plt.savefig(f"All-Plots/{model_name}/heatmap.jpg")
    plt.clf()

    X = np.linspace(extent[0], extent[1], n_bins)
    Y = np.linspace(extent[2], extent[3], n_bins)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, hmp, 100, color='binary')
    ax.set_xlabel("Impulse")
    ax.set_ylabel("Angle (pi radians)")
    ax.set_zlabel("Density")
    plt.savefig(f"All-Plots/{model_name}/contour.jpg")
    plt.clf()


def plot_action_space_usage(model_name, assay_config, assay_id, n, impulse_scaling, angle_scaling, abs_angles=True):
    if not os.path.exists(f"All-Plots/{model_name}/"):
        os.makedirs(f"All-Plots/{model_name}/")

    all_impulses, all_angles, consumption_timestamps, predation_sequences, predator_presence = \
        get_multiple_actions(model_name, assay_config, assay_id, n)
    mu_impulse, mu_angle = get_multiple_means(model_name, assay_config, assay_id, n)

    mu_impulse *= impulse_scaling
    angle_scaling *= angle_scaling

    if abs_angles:
        all_angles = np.absolute(all_angles)
        mu_angle = np.absolute(mu_angle)

    plot_action_scatter(all_impulses, all_angles, model_name)
    plot_action_use_density(mu_impulse, mu_angle, model_name)


    mu_impulse = [i * impulse_scaling for i in mu_impulse]
    mu_angle = [i * angle_scaling for i in mu_angle]
    consumption_mu_imp = [a for i, a in enumerate(mu_impulse) if i in consumption_timestamps]
    consumption_mu_ang = [a for i, a in enumerate(mu_angle) if i in consumption_timestamps]
    predator_mu_imp = [a for i, a in enumerate(mu_impulse) if i in predator_presence]
    predator_mu_ang = [a for i, a in enumerate(mu_angle) if i in predator_presence]

    plot_action_scatter(mu_impulse, mu_angle, model_name, [consumption_mu_imp, predator_mu_imp],
                        [consumption_mu_ang, predator_mu_ang], ["Consumption", "Predator Present"])

    consumption_mu_imp = [a for i, a in enumerate(mu_impulse) if i in predation_sequences]
    consumption_mu_ang = [a for i, a in enumerate(mu_angle) if i in predation_sequences]
    plot_action_scatter(mu_impulse, mu_angle, model_name, [consumption_mu_imp],
                        [consumption_mu_ang], ["Prey Capture"])


def plot_actions_under_all_contexts(model_name, assay_config, assay_id, n, impulse_scaling, angle_scaling, abs_angles=True):
    if not os.path.exists(f"All-Plots/{model_name}/"):
        os.makedirs(f"All-Plots/{model_name}/")

    datas = []
    for i in range(1, n+1):
        datas.append(load_data(model_name, assay_config, f"{assay_id}-{i}"))

    context_labels = label_behavioural_context_multiple_trials(datas, model_name)
    mu_impulse, mu_angle = get_multiple_means(model_name, assay_config, assay_id, n)

    if abs_angles:
        mu_angle = np.absolute(mu_angle)

    mu_impulse *= impulse_scaling
    mu_angle *= angle_scaling

    all_contexts = np.concatenate((context_labels), axis=0)

    for context in range(all_contexts.shape[1]):
        important_points = [i for i, v in enumerate(all_contexts[:, context]) if v == 1]
        mu_impulse_s = mu_impulse[important_points]
        mu_angle_s = mu_angle[important_points]

        plot_action_scatter(mu_impulse, mu_angle, model_name, [mu_impulse_s], [mu_angle_s],
                            [get_behavioural_context_name_by_index(context)])


if __name__ == "__main__":
    model_name = "ppo_scaffold_21-2"

    max_impulse = 16
    angle_scaling = 1

    plot_action_space_usage(model_name, "Behavioural-Data-Free", "Naturalistic", 20, max_impulse, angle_scaling)
    plot_actions_under_all_contexts(model_name, "Behavioural-Data-Free", "Naturalistic", 20, max_impulse, angle_scaling)




