import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

from Analysis.load_data import load_data
from Analysis.Behavioural.Continuous.plot_behavioural_choice import get_multiple_means
from Analysis.Behavioural.Continuous.identifying_bouts import cluster_bouts
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def draw_fish(x, y, mouth_size, head_size, tail_length, ax):
    mouth_centre = (x, y)  # TODO: Make sure this is true

    mouth = plt.Circle(mouth_centre, mouth_size, fc="green")
    ax.add_patch(mouth)

    angle = (2 * np.pi) / 2
    dx1, dy1 = head_size * np.sin(angle), head_size * np.cos(angle)
    head_centre = (mouth_centre[0] + dx1,
                   mouth_centre[1] + dy1)
    head = plt.Circle(head_centre, head_size, fc="green")
    ax.add_patch(head)

    dx2, dy2 = -1 * dy1, dx1
    left_flank = (head_centre[0] + dx2,
                  head_centre[1] + dy2)
    right_flank = (head_centre[0] - dx2,
                   head_centre[1] - dy2)
    tip = (mouth_centre[0] + (tail_length + head_size) * np.sin(angle),
           mouth_centre[1] + (tail_length + head_size) * np.cos(angle))
    tail = plt.Polygon(np.array([left_flank, right_flank, tip]), fc="green")
    ax.add_patch(tail)
    return ax


def create_density_cloud(density_list, action_num, stimulus_name, return_objects, model_name, assay_config,
                         steps_prior):
    n_samples = len(density_list)
    x = np.array([i[0] for i in density_list])
    y = np.array([i[1] for i in density_list])
    y = np.negative(y)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="Reds", )  # cmap='gist_gray')#  cmap='PuBu_r')
    fig.colorbar(pcm)

    scale_bar = AnchoredSizeBar(ax.transData,
                                200, '20mm', 'lower right',
                                pad=1,
                                color='black',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 16}
                                )
    ax.add_artist(scale_bar)
    # plt.arrow(-300, 220, 0, 40, width=10, color="red")

    ax = draw_fish(-300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(-600, 0)
    ax.set_ylim(-80, 520)

    # plt.scatter(y, x, c="b", s=1)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: {stimulus_name}, Action: {action_num}, N-Samples: {n_samples}")
    plt.savefig(f"All-Plots/{model_name}/{assay_config}-{stimulus_name}-{action_num}-steps_prior: {steps_prior}.jpg")

    if return_objects:
        plt.clf()
        plt.close()
        return ax
    else:
        plt.clf()
        plt.close()


def create_density_cloud_overlap(cloud_left, cloud_right, action_num, stimulus_name, return_objects, model_name,
                                 assay_config, steps_prior, normalise_laterality):
    n_samples = len(cloud_left) + len(cloud_right)
    # For left
    x = np.array([i[0] for i in cloud_left])
    y = np.array([i[1] for i in cloud_left])
    # y = np.negative(y)
    nbins = 300
    try:
        k = kde.gaussian_kde([y, x])
    except ValueError:
        return
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # For right
    x = np.array([i[0] for i in cloud_right])
    y = np.array([i[1] for i in cloud_right])
    # y = np.negative(y)
    nbins = 300
    try:
        k = kde.gaussian_kde([y, x])
    except ValueError:
        return
    zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))

    if normalise_laterality:
        zi = zi/np.sum(zi)
        zi2 = zi2/np.sum(zi2)

    zi = zi - zi2

    # Make the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    zi = np.clip(zi, -0.0000015, 0.0000015)  # TODO: Remove

    pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdBu')

    # ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
    #                        pad=0.6, sep=4, linekw=dict(color="crimson"), )

    scale_bar = AnchoredSizeBar(ax.transData,
                                200, '20mm', 'lower right',
                                pad=1,
                                color='black',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 16}
                                )
    ax.add_artist(scale_bar)

    ax = draw_fish(300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(0, 600)
    ax.set_ylim(-80, 520)

    fig.colorbar(pcm)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: {stimulus_name}, Action: {action_num}, N-Samples: {n_samples}")
    plt.savefig(f"{model_name}/{assay_config}-{stimulus_name}-{action_num}-steps_prior: {steps_prior}.jpg")

    if return_objects:
        plt.clf()
        plt.close()
        return ax
    else:
        plt.clf()
        plt.close()


def get_nearby_features(data, step, proximity=300):
    """For a single step, returns the positions of nearby prey and predators."""
    nearby_prey_coordinates = []
    nearby_predator_coordinates = []
    nearby_area = [[data["fish_position"][step][0] - proximity,
                    data["fish_position"][step][0] + proximity],
                   [data["fish_position"][step][1] - proximity,
                    data["fish_position"][step][1] + proximity]
                   ]

    for i in data["prey_positions"][step]:
        is_in_area = nearby_area[0][0] <= i[0] <= nearby_area[0][1] and \
                     nearby_area[1][0] <= i[1] <= nearby_area[1][1]
        if is_in_area:
            nearby_prey_coordinates.append(i)
    j = data["predator_positions"][step]
    is_in_area = nearby_area[0][0] <= j[0] <= nearby_area[0][1] and \
                 nearby_area[1][0] <= j[1] <= nearby_area[1][1]
    if is_in_area:
        nearby_predator_coordinates.append(j)
    return nearby_prey_coordinates, nearby_predator_coordinates


def transform_to_egocentric(feature_positions, fish_position, fish_orientation):
    """Takes the feature coordinates and fish position and translates them onto an egocentric reference frame."""
    transformed_coordinates = []
    for i in feature_positions:
        v = [i[0] - fish_position[0], i[1] - fish_position[1]]
        theta = 2 * np.pi - fish_orientation
        tran_v = [np.cos(theta) * v[0] - np.sin(theta) * v[1],
                  np.sin(theta) * v[0] + np.cos(theta) * v[1]]
        tran_v[0] = tran_v[0] + 300
        tran_v[1] = tran_v[1] + 300
        transformed_coordinates.append(tran_v)
    return transformed_coordinates


def get_clouds_with_action(data, action, predictor, steps_prior=0, direction=None, labels=None):
    prey_cloud = []
    predator_cloud = []
    if labels is not None:
        all_actions = labels
    else:
        all_actions = predictor.predict(
            [[imp, ang] for imp, ang in zip(data["mu_impulse"][:, 0], np.absolute(data["mu_angle"][:, 0]))])
    all_actions = all_actions[1:]

    if direction == "Left":
        angle_correctness = (data["mu_angle"][:, 0] > 0)
    elif direction == "Right":
        angle_correctness = (data["mu_angle"][:, 0] < 0)
    else:
        angle_correctness = data["mu_angle"][:, 0] * True

    for i, a in enumerate(all_actions):
        if i - steps_prior >= 0:
            if a == action and angle_correctness[i]:
                allocentric_prey, allocentric_predators = get_nearby_features(data, i - steps_prior)

                if len(allocentric_prey) > 0:
                    egocentric_prey = transform_to_egocentric(allocentric_prey, data["fish_position"][i - steps_prior],
                                                              data["fish_angle"][i - steps_prior])
                else:
                    egocentric_prey = []

                if len(allocentric_predators) > 0:
                    egocentric_predators = transform_to_egocentric(allocentric_predators,
                                                                   data["fish_position"][i - steps_prior],
                                                                   data["fish_angle"][i - steps_prior])
                else:
                    egocentric_predators = []

                prey_cloud = prey_cloud + egocentric_prey
                predator_cloud = predator_cloud + egocentric_predators

    return prey_cloud, predator_cloud


def get_all_density_plots_all_subsets_continuous(model_name, assay_config, assay_id, n, impulse_scaling, angle_scaling,
                                                 return_objects=False, steps_prior=0, n_clusters=5,
                                                 threshold_for_laterality=0.1, normalise_laterality=True,
                                                 cluster_algo="KNN"):
    if not os.path.exists(f"{model_name}/"):
        os.makedirs(f"{model_name}/")

    mu_impulse, mu_angle = get_multiple_means(model_name, assay_config, assay_id, n)
    predictor = cluster_bouts(mu_impulse, mu_angle, cluster_algo, n_clusters, model_name, impulse_scaling, angle_scaling)

    all_actions = predictor.labels_
    action_mean_angle = []

    n_clusters = len(set(all_actions))
    for c in range(n_clusters):
        action_mean_angle.append(np.mean(np.absolute(mu_angle[all_actions == c])))
    action_mean_angle = np.array(action_mean_angle)
    lateralised_bout = action_mean_angle > threshold_for_laterality

    axes_objects = []

    for action_num in range(0, n_clusters):
        prey_cloud = []
        pred_cloud = []

        prey_cloud_left = []
        pred_cloud_left = []
        prey_cloud_right = []
        pred_cloud_right = []

        current_index = 0

        for i in range(1, n + 1):
            if i > 100:
                data = load_data(f"{model_name}", f"{assay_config}", f"{assay_id}-{i}")
            else:
                data = load_data(f"{model_name}", f"{assay_config}", f"{assay_id}-{i}")
            existing_labels = predictor.labels_[current_index: current_index+data["observation"].shape[0]]

            if lateralised_bout[action_num]:
                prey_left, pred_left = get_clouds_with_action(data, action_num, predictor, direction="Left", labels=existing_labels)
                prey_right, pred_right = get_clouds_with_action(data, action_num, predictor, direction="Right", labels=existing_labels)

                prey_cloud_left = prey_cloud_left + prey_left
                pred_cloud_left = pred_cloud_left + pred_left

                prey_cloud_right = prey_cloud_right + prey_right
                pred_cloud_right = pred_cloud_right + pred_right
            else:
                prey, pred = get_clouds_with_action(data, action_num, predictor, labels=existing_labels)
                prey_cloud = prey_cloud + prey
                pred_cloud = pred_cloud + pred
            current_index += data["observation"].shape[0]

        if lateralised_bout[action_num]:
            if len(prey_cloud_left) > 2:
                ax = create_density_cloud(prey_cloud_left, str(action_num) + "-Left", "Prey", return_objects,
                                          save_location=model_name,
                                          assay_config=assay_config, steps_prior=steps_prior)
                axes_objects.append(ax)
            if len(prey_cloud_right) > 2:
                ax = create_density_cloud(prey_cloud_right, str(action_num) + "-Right", "Prey", return_objects,
                                          save_location=model_name,
                                          assay_config=assay_config, steps_prior=steps_prior)
                axes_objects.append(ax)

            if len(prey_cloud_left) > 2 and len(prey_cloud_right) > 2:
                create_density_cloud_overlap(prey_cloud_left, prey_cloud_right, str(action_num), "Prey", return_objects,
                                             model_name=model_name, assay_config=assay_config, steps_prior=steps_prior,
                                             normalise_laterality=normalise_laterality)

            if len(pred_cloud_left) > 2:
                ax = create_density_cloud(pred_cloud_left, str(action_num) + "-Left", "Predator", return_objects,
                                          save_location=model_name,
                                          assay_config=assay_config, steps_prior=steps_prior)
                axes_objects.append(ax)
            if len(pred_cloud_right) > 2:
                ax = create_density_cloud(pred_cloud_right, str(action_num) + "-Right", "Predator", return_objects,
                                          save_location=model_name,
                                          assay_config=assay_config, steps_prior=steps_prior)
                axes_objects.append(ax)

            if len(pred_cloud_left) > 2 and len(pred_cloud_right) > 2:
                create_density_cloud_overlap(pred_cloud_left, pred_cloud_right, str(action_num), "Predator",
                                             return_objects, model_name=model_name,
                                             assay_config=assay_config, steps_prior=steps_prior,
                                             normalise_laterality=normalise_laterality)

        else:
            if len(prey_cloud) > 2:
                ax = create_density_cloud(prey_cloud, action_num, "Prey", return_objects, save_location=model_name,
                                          assay_config=assay_config, steps_prior=steps_prior)
                axes_objects.append(ax)

            if len(pred_cloud) > 2:
                ax = create_density_cloud(pred_cloud, action_num, "Predator", return_objects, save_location=model_name,
                                          assay_config=assay_config, steps_prior=steps_prior)
                axes_objects.append(ax)


if __name__ == "__main__":
    model_name, assay_config, assay_id, n = "ppo_scaffold_21-2", "Behavioural-Data-Free", "Naturalistic", 20
    get_all_density_plots_all_subsets_continuous(model_name, assay_config, assay_id, n, impulse_scaling=16,
                                                 angle_scaling=1, cluster_algo="AGG")
