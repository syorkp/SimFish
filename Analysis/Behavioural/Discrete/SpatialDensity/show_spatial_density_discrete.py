import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import os

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn import linear_model

from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.get_action_name import get_action_name
from Analysis.Behavioural.Tools.anchored_scale_bar import AnchoredHScaleBar

"""
To create a graph of the style in Figure 3b of Marques et al. (2018)
"""


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


def get_nearby_features(data, step, proximity=500):
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


def get_nearby_features_predictive(data, step, proximity=300, steps_used_in_prediction=10):
    """For a single step, returns the positions of nearby prey and predators."""
    if step < steps_used_in_prediction:
        return [], []

    # Define nearby area
    nearby_area = [[data["fish_position"][step][0] - proximity,
                    data["fish_position"][step][0] + proximity],
                   [data["fish_position"][step][1] - proximity,
                    data["fish_position"][step][1] + proximity]
                   ]

    # Whether prey are in area
    prey_in_area = (data["prey_positions"][step, :, 0] >= nearby_area[0][0] * 1) * \
                    (data["prey_positions"][step, :, 0] <= nearby_area[0][1] * 1) * \
                     (data["prey_positions"][step, :, 1] >= nearby_area[1][0] * 1) * \
                      (data["prey_positions"][step, :, 1] <= nearby_area[1][1] * 1)

    previous_prey_coordinates = data["prey_positions"][step-steps_used_in_prediction:step+1, prey_in_area]
    # Build linear model on all prey data, with X being the previous positions (t-10) and Y being [x, y]
    predicted_prey_coordinates = []
    for p in range(previous_prey_coordinates.shape[1]):
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(previous_prey_coordinates[:-1, p], previous_prey_coordinates[1:, p])
        next = regr.predict(previous_prey_coordinates[-1:, p])
        predicted_prey_coordinates.append(next[0])
    predicted_prey_coordinates = np.array(predicted_prey_coordinates)

    predator_in_area = nearby_area[0][0] <= data["predator_positions"][step][0] <= nearby_area[0][1] and \
                       nearby_area[1][0] <= data["predator_positions"][step][1] <= nearby_area[1][1]
    nearby_predator_coordinates = data["predator_positions"][step][predator_in_area]
    previous_predator_coordinates = data["predator_positions"][step-steps_used_in_prediction:step][predator_in_area]

    # Extra step for weird predictions which result in very high or low values. TODO: Instead clip based on available range (slightly extended)
    if len(predicted_prey_coordinates) > 0:
        to_remove_prey = (predicted_prey_coordinates < 0) + (predicted_prey_coordinates > 3000)
        to_remove_prey = to_remove_prey[:, 0] + to_remove_prey[:, 1]
        if len(predicted_prey_coordinates) > 0:
            x = True
        predicted_prey_coordinates = predicted_prey_coordinates[~to_remove_prey]

    # TODO: DO same for Pred
    to_remove_predators = (nearby_predator_coordinates < 0) + (nearby_predator_coordinates > 1000)
    nearby_predator_coordinates = nearby_predator_coordinates[~to_remove_predators]

    return list(predicted_prey_coordinates), list(nearby_predator_coordinates)


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


def get_clouds_with_action(data, action=0, steps_prior=0, predictive=False):
    prey_cloud = []
    predator_cloud = []

    for i, step in enumerate(data["action"]):
        if i - steps_prior >= 0:
            if data["action"][i] == action:
                if predictive:
                    allocentric_prey, allocentric_predators = get_nearby_features_predictive(data, i - steps_prior)

                else:
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


def create_density_cloud(density_list, action_num, stimulus_name, return_objects, save_location, assay_config,
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
    plt.title(f"Feature: {stimulus_name}, Action: {get_action_name(action_num)}, N-Samples: {n_samples}")
    plt.savefig(
        f"{save_location}{assay_config}-{stimulus_name}-{get_action_name(action_num)}-steps_prior: {steps_prior}.jpg",
        bbox_inches='tight')

    if return_objects:
        plt.clf()
        plt.close()
        return ax
    else:
        plt.clf()
        plt.close()


def create_density_cloud_compared(density_list1, density_list2, action_num, stimulus_name, return_objects,
                                  save_location, assay_config):
    nbins = 300
    # Pre-transition
    n_samples1 = len(density_list1)
    x1 = np.array([i[0] for i in density_list1])
    y1 = np.array([i[1] for i in density_list1])
    y1 = np.negative(y1)

    # Post-transition
    n_samples2 = len(density_list2)
    x2 = np.array([i[0] for i in density_list2])
    y2 = np.array([i[1] for i in density_list2])
    y2 = np.negative(y2)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

    k1 = kde.gaussian_kde([y1, x1])
    yi, xi = np.mgrid[min(x1.min(), x2.min()):max(x1.max(), x2.max()):nbins * 1j,
             min(y1.min(), y2.min()):max(y1.max(), y2.min()):nbins * 1j]
    zi1 = k1(np.vstack([xi.flatten(), yi.flatten()]))
    zi1 = zi1 / np.sum(zi1)

    k2 = kde.gaussian_kde([y2, x2])
    zi2 = k2(np.vstack([xi.flatten(), yi.flatten()]))  # /n_samples2
    zi2 = zi2 / np.sum(zi2)

    zi = zi2 - zi1
    z_max, z_min = np.max(zi), np.min(zi)
    half_span = max(abs(z_max), abs(z_min))

    # Make the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="coolwarm", vmin=-half_span,
                        vmax=half_span)  # cmap='gist_gray')#  cmap='PuBu_r')
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
    plt.title(f"Feature: {stimulus_name}, Action: {get_action_name(action_num)}, N-Samples: {n_samples1}, {n_samples2}")
    plt.savefig(f"{save_location}{assay_config}-{stimulus_name}-{get_action_name(action_num)}.jpg")

    if return_objects:
        plt.clf()
        plt.close()
        return ax
    else:
        plt.clf()
        plt.close()


def get_all_density_plots(data, save_location):
    for action_num in range(0, 12):
        prey_1, pred_1 = get_clouds_with_action(data, action_num)

        if len(prey_1) > 2:
            create_density_cloud(prey_1, action_num, "Prey", False, save_location, "Training", 0)

        if len(pred_1) > 2:
            create_density_cloud(pred_1, action_num, "Predator", False, save_location, "Training", 0)


def get_all_density_plots_all_subsets(p1, p2, p3, n, return_objects, steps_prior=0, position_predictive=False):
    if not os.path.exists(f"../../../../Analysis-Output/Behavioural/Spatial-Density-Plots/{p1}/"):
        os.makedirs(f"../../../../Analysis-Output/Behavioural/Spatial-Density-Plots/{p1}/")
    save_location = f"../../../../Analysis-Output/Behavioural/Spatial-Density-Plots/{p1}/"
    if position_predictive:
        save_location += "Predictive-"

    axes_objects = []
    for action_num in range(0, 12):

        prey_cloud = []
        pred_cloud = []
        for i in range(1, n + 1):
            print(i)
            if i > 100:
                data = load_data(p1, f"{p2}-2", f"{p3} {i}")
            else:
                data = load_data(p1, p2, f"{p3}-{i}")

            prey_1, pred_1 = get_clouds_with_action(data, action_num, steps_prior, predictive=position_predictive)
            prey_cloud = prey_cloud + prey_1
            pred_cloud = pred_cloud + pred_1

        if len(prey_cloud) > 2:
            ax = create_density_cloud(prey_cloud, action_num, "Prey", return_objects, save_location=save_location, assay_config=p2,
                                      steps_prior=steps_prior)
            axes_objects.append(ax)

        if len(pred_cloud) > 2:
            ax = create_density_cloud(pred_cloud, action_num, "Predator", return_objects, save_location=save_location,
                                      assay_config=p2, steps_prior=steps_prior)
            axes_objects.append(ax)

    ax = create_j_turn_overlap_plot(p1, p2, p3, n, return_objects, save_location=save_location)
    axes_objects.append(ax)
    ax = create_j_turn_2_overlap_plot(p1, p2, p3, n, return_objects, save_location=save_location)
    axes_objects.append(ax)
    ax = create_routine_turn_overlap_plot(p1, p2, p3, n, return_objects, save_location=save_location)
    axes_objects.append(ax)
    ax = create_cstart_overlap_plot(p1, p2, p3, n, return_objects, save_location=save_location)
    axes_objects.append(ax)
    return axes_objects


def create_overlap_plot(cloud_left, cloud_right, stimulus_name, action, model_name, assay_config, save_location):
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
    zi = zi2 - zi

    # LEFT IS BLUE, RIGHT IS RED

    # Make the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    # zi = np.clip(zi, -0.0000015, 0.0000015)  # TODO: Remove

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
    plt.title(f"Feature: {stimulus_name}, Action: {action}, N-Samples: {n_samples}")
    plt.savefig(f"{save_location}{stimulus_name}-{action}.jpg", bbox_inches='tight')
    plt.clf()
    plt.close()


def create_j_turn_overlap_plot(p1, p2, p3, n, return_objects, save_location):
    prey_cloud_left = []
    pred_cloud_left = []
    for i in range(1, n + 1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 4)
        prey_cloud_left = prey_cloud_left + prey_1
        pred_cloud_left = pred_cloud_left + pred_1
    prey_cloud_right = []
    pred_cloud_right = []
    for i in range(1, n + 1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 5)
        prey_cloud_right = prey_cloud_right + prey_1
        pred_cloud_right = pred_cloud_right + pred_1

    create_overlap_plot(prey_cloud_left, prey_cloud_right, "Prey", "J-turn", p1, assay_config=p2, save_location=save_location)
    create_overlap_plot(pred_cloud_left, pred_cloud_right, "Predators", "J-turn", p1, assay_config=p2, save_location=save_location)

    if return_objects:
        plt.clf()
        return None
    else:
        return


def create_j_turn_2_overlap_plot(p1, p2, p3, n, return_objects, save_location):
    prey_cloud_left = []
    pred_cloud_left = []
    for i in range(1, n + 1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 10)
        prey_cloud_left = prey_cloud_left + prey_1
        pred_cloud_left = pred_cloud_left + pred_1
    prey_cloud_right = []
    pred_cloud_right = []
    for i in range(1, n + 1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 11)
        prey_cloud_right = prey_cloud_right + prey_1
        pred_cloud_right = pred_cloud_right + pred_1

    create_overlap_plot(prey_cloud_left, prey_cloud_right, "Prey", "J-turn 2", p1, assay_config=p2, save_location=save_location)
    create_overlap_plot(pred_cloud_left, pred_cloud_right, "Predators", "J-turn 2", p1, assay_config=p2, save_location=save_location)

    if return_objects:
        plt.clf()
        return None
    else:
        return


def create_routine_turn_overlap_plot(p1, p2, p3, n, return_objects, save_location):
    prey_cloud_left = []
    pred_cloud_left = []
    for i in range(1, n + 1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 1)
        prey_cloud_left = prey_cloud_left + prey_1
        pred_cloud_left = pred_cloud_left + pred_1

    prey_cloud_right = []
    pred_cloud_right = []
    for i in range(1, n + 1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 2)
        prey_cloud_right = prey_cloud_right + prey_1
        pred_cloud_right = pred_cloud_right + pred_1

    create_overlap_plot(prey_cloud_left, prey_cloud_right, "Prey", "RT", p1, assay_config=p2, save_location=save_location)
    create_overlap_plot(pred_cloud_left, pred_cloud_right, "Predators", "RT", p1, assay_config=p2, save_location=save_location)

    if return_objects:
        plt.clf()
        return None
    else:
        return

    # prey_cloud_left = []
    # for i in range(1, n+1):
    #     data = load_data(p1, p2, f"{p3}-{i}")
    #     prey_1, pred_1 = get_clouds_with_action(data, 1)
    #     prey_cloud_left = prey_cloud_left + prey_1
    # prey_cloud_right = []
    # for i in range(1, n+1):
    #     data = load_data(p1, p2, f"{p3}-{i}")
    #     prey_1, pred_1 = get_clouds_with_action(data, 2)
    #     prey_cloud_right = prey_cloud_right + prey_1
    # n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # # For left
    # x = np.array([i[0] for i in prey_cloud_left])
    # y = np.array([i[1] for i in prey_cloud_left])
    # #y = np.negative(y)
    # nbins = 300
    # try:
    #     k = kde.gaussian_kde([y, x])
    # except ValueError:
    #     return
    # yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    #
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #
    # # For right
    # x = np.array([i[0] for i in prey_cloud_right])
    # y = np.array([i[1] for i in prey_cloud_right])
    # #y = np.negative(y)
    # nbins = 300
    # try:
    #     k = kde.gaussian_kde([y, x])
    # except ValueError:
    #     return
    # zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))
    #
    # zi = zi2 - zi
    #
    # fig, ax = plt.subplots()
    #
    # pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdBu')
    #
    # ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
    #                        pad=0.6, sep=4, linekw=dict(color="crimson"), )
    # ax.add_artist(ob)
    #
    #
    # ax = draw_fish(300, 220, 4, 2.5, 41.5, ax)
    # ax.set_xlim(0, 600)
    # ax.set_ylim(-80, 520)
    # # ax = draw_fish(-300, 220, 4, 2.5, 41.5, ax)
    # # ax.set_xlim(-600, 0)
    # # ax.set_ylim(-80, 520)
    #
    # fig.colorbar(pcm)
    #
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # plt.title(f"Feature: Prey, Action: Routine turns, N-Samples: {n_samples}")
    #
    # if return_objects:
    #     plt.clf()
    #     return ax
    # else:
    #     plt.show()


def create_cstart_overlap_plot(p1, p2, p3, n, return_objects, save_location):
    prey_cloud_left = []
    pred_cloud_left = []
    for i in range(1, n + 1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 8)
        prey_cloud_left = prey_cloud_left + prey_1
        pred_cloud_left = pred_cloud_left + pred_1
    prey_cloud_right = []
    pred_cloud_right = []
    for i in range(1, n + 1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 7)
        prey_cloud_right = prey_cloud_right + prey_1
        pred_cloud_right = pred_cloud_right + pred_1

    create_overlap_plot(prey_cloud_left, prey_cloud_right, "Prey", "C-Start", p1, assay_config=p2, save_location=save_location)
    create_overlap_plot(pred_cloud_left, pred_cloud_right, "Predators", "C-Start", p1, assay_config=p2, save_location=save_location)

    if return_objects:
        plt.clf()
        return None
    else:
        return

    #
    # prey_cloud_left = []
    # for i in range(1, n+1):
    #     if i < 11:
    #         data = load_data(p1, f"{p2}-2", f"{p3}-{i}")
    #     else:
    #         print(i)
    #         data = load_data(p1, p2, f"{p3}-{i}")
    #     prey_1, pred_1 = get_clouds_with_action(data, 7)
    #     prey_cloud_left = prey_cloud_left + prey_1
    # prey_cloud_right = []
    # for i in range(1, n+1):
    #     if i < 11: continue
    #         # data = load_data(p1, f"{p2}-2", f"{p3} {i}")
    #     else:
    #         data = load_data(p1, p2, f"{p3} {i}")
    #     prey_1, pred_1 = get_clouds_with_action(data, 8)
    #     prey_cloud_right = prey_cloud_right + prey_1
    # n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # # For left
    # x = np.array([i[0] for i in prey_cloud_left])
    # y = np.array([i[1] for i in prey_cloud_left])
    # #y = np.negative(y)
    # nbins = 300
    # try:
    #     k = kde.gaussian_kde([y, x])
    # except ValueError:
    #     return
    # yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    #
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #
    # # For right
    # x = np.array([i[0] for i in prey_cloud_right])
    # y = np.array([i[1] for i in prey_cloud_right])
    # #y = np.negative(y)
    # nbins = 300
    # try:
    #     k = kde.gaussian_kde([y, x])
    # except ValueError:
    #     return
    # zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))
    #
    # zi = zi - zi2
    # # Make the plot
    # fig, ax = plt.subplots()
    #
    # pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdBu')
    #
    # ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
    #                        pad=0.6, sep=4, linekw=dict(color="crimson"), )
    # ax.add_artist(ob)
    #
    # ax = draw_fish(300, 220, 4, 2.5, 41.5, ax)
    # ax.set_xlim(0, 600)
    # ax.set_ylim(-80, 520)
    #
    # fig.colorbar(pcm)
    #
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # plt.title(f"Feature: Predator, Action: C-Starts, N-Samples: {n_samples}")
    #
    # if return_objects:
    #     plt.clf()
    #     return ax
    # else:
    #     plt.show()


def create_j_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2):
    prey_cloud_left = []
    prey_cloud_right = []
    for m in range(1, n2 + 1):
        for i in range(1, n + 1):
            data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")
            prey_1, pred_1 = get_clouds_with_action(data, 5)
            prey_cloud_left = prey_cloud_left + prey_1
        for i in range(1, n + 1):
            data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")
            prey_1, pred_1 = get_clouds_with_action(data, 4)
            prey_cloud_right = prey_cloud_right + prey_1
    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    # y = np.negative(y)
    nbins = 300
    try:
        k = kde.gaussian_kde([y, x])
    except ValueError:
        return
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # For right
    x = np.array([i[0] for i in prey_cloud_right])
    y = np.array([i[1] for i in prey_cloud_right])
    # y = np.negative(y)
    nbins = 300
    k = kde.gaussian_kde([y, x])
    zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi - zi2

    # Make the plot
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdBu')

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)

    ax = draw_fish(300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(0, 600)
    ax.set_ylim(-80, 520)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Prey, Action: J-turns")
    plt.show()


def create_routine_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2):
    prey_cloud_left = []
    prey_cloud_right = []

    for m in range(1, n2 + 1):
        for i in range(1, n + 1):
            data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")
            prey_1, pred_1 = get_clouds_with_action(data, 2)
            prey_cloud_left = prey_cloud_left + prey_1
        for i in range(1, n + 1):
            data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")
            prey_1, pred_1 = get_clouds_with_action(data, 1)
            prey_cloud_right = prey_cloud_right + prey_1

    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    # y = np.negative(y)
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # For right
    x = np.array([i[0] for i in prey_cloud_right])
    y = np.array([i[1] for i in prey_cloud_right])
    # y = np.negative(y)
    nbins = 300
    k = kde.gaussian_kde([y, x])
    zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))

    zi = zi2 - zi

    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdBu')

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)

    ax = draw_fish(300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(0, 600)
    ax.set_ylim(-80, 520)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Prey, Action: Routine turns")
    plt.show()


def create_cstart_overlap_plot_multiple_models(p1, p2, p3, n, n2):
    prey_cloud_left = []
    prey_cloud_right = []
    for m in range(1, n2 + 1):

        for i in range(1, n + 1):
            if i < 11:
                continue
            # print(i)
            #
            # data = load_data(p1, f"{p2}-2", f"{p3}-{i}")
            else:
                print(i)
                data = load_data(f"{p1}-{m}", p2, f"{p3} {i}")
            prey_1, pred_1 = get_clouds_with_action(data, 8)
            prey_cloud_left = prey_cloud_left + prey_1
        for i in range(1, n + 1):
            if i < 11:
                continue
            # data = load_data(p1, f"{p2}-2", f"{p3} {i}")
            else:
                data = load_data(f"{p1}-{m}", p2, f"{p3} {i}")
            prey_1, pred_1 = get_clouds_with_action(data, 7)
            prey_cloud_right = prey_cloud_right + prey_1
    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    # y = np.negative(y)
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # For right
    x = np.array([i[0] for i in prey_cloud_right])
    y = np.array([i[1] for i in prey_cloud_right])
    # y = np.negative(y)
    nbins = 300
    k = kde.gaussian_kde([y, x])
    zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))

    zi = zi - zi2
    # Make the plot
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdBu')

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)

    ax = draw_fish(300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(0, 600)
    ax.set_ylim(-80, 520)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Predator, Action: C-Starts")
    plt.show()


def get_all_density_plots_multiple_models(p1, p2, p3, n, n2, steps_prior=0):
    for action_num in range(0, 10):
        prey_cloud = []
        pred_cloud = []
        for m in range(1, n2 + 1):
            for i in range(1, n + 1):
                if i > 12:
                    data = load_data(f"{p1}-{m}", f"{p2}-2", f"{p3} {i}")
                else:
                    data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")

                prey_1, pred_1 = get_clouds_with_action(data, action_num, steps_prior)
                prey_cloud = prey_cloud + prey_1
                pred_cloud = pred_cloud + pred_1

        if len(prey_cloud) > 2:
            create_density_cloud(prey_cloud, action_num, "Prey", False, p1, p2)

        if len(pred_cloud) > 2:
            create_density_cloud(pred_cloud, action_num, "Predator", False, p1, p2)

    create_j_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2)
    create_routine_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2)
    create_cstart_overlap_plot_multiple_models(p1, p2, p3, n, n2)


def plot_all_density_plots_across_scaffold(model_name, assay_config_a, assay_config_b, assay_id, n):
    for action_num in range(0, 10):
        prey_cloud1 = []
        pred_cloud1 = []
        prey_cloud2 = []
        pred_cloud2 = []
        for i in range(1, n + 1):
            if i > 100:
                data1 = load_data(f"{model_name}", f"{assay_config_a}", f"{assay_id}-{i}")
                data2 = load_data(f"{model_name}", f"{assay_config_b}", f"{assay_id}-{i}")
            else:
                data1 = load_data(f"{model_name}", f"{assay_config_a}", f"{assay_id}-{i}")
                data2 = load_data(f"{model_name}", f"{assay_config_b}", f"{assay_id}-{i}")

            prey_1, pred_1 = get_clouds_with_action(data1, action_num)
            prey_2, pred_2 = get_clouds_with_action(data2, action_num)
            prey_cloud1 = prey_cloud1 + prey_1
            pred_cloud1 = pred_cloud1 + pred_1
            prey_cloud2 = prey_cloud2 + prey_2
            pred_cloud2 = pred_cloud2 + pred_2

        if len(prey_cloud1) > 2 and len(prey_cloud2) > 2:
            create_density_cloud_compared(prey_cloud1, prey_cloud2, action_num, "Prey", False, model_name,
                                          f"{assay_config_a}-{assay_config_b}")

        if len(pred_cloud1) > 2 and len(pred_cloud2) > 2:
            create_density_cloud_compared(pred_cloud1, pred_cloud2, action_num, "Predator", False, model_name,
                                          f"{assay_config_a}-{assay_config_b}")

    # create_j_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2)
    # create_routine_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2)
    # create_cstart_overlap_plot_multiple_models(p1, p2, p3, n, n2)


if __name__ == "__main__":
    # VERSION 2, 2023
    # get_all_density_plots_all_subsets(f"dqn_gamma_pm-4", "Behavioural-Data-Free", "Naturalistic", 100, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic", 50, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_gamma-3", "Behavioural-Data-Free", "Naturalistic", 50, return_objects=False)
    get_all_density_plots_all_subsets(f"dqn_0-1", "Behavioural-Data-Free", "Naturalistic", 40, return_objects=False)
    get_all_density_plots_all_subsets(f"dqn_0-7", "Behavioural-Data-Free", "Naturalistic", 40, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_0-5", "Behavioural-Data-Free", "Naturalistic", 100, return_objects=False)

    # d = load_data("dqn_0-1", "Episode 800", "Episode 800", training_data=True)
    # save_location = f"../../../../Analysis-Output/Behavioural/Spatial-Density-Plots/dqn_0-1/"
    # get_all_density_plots(d, save_location)
    #
    # create_cstart_overlap_plot(f"dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100, return_objects=False,
    #                            save_location=save_location)
    # create_routine_turn_overlap_plot(f"dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100, return_objects=False,
    #                            save_location=save_location)
    # create_j_turn_overlap_plot(f"dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100, return_objects=False,
    #                            save_location=save_location)
    # create_j_turn_2_overlap_plot(f"dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100, return_objects=False,
    #                            save_location=save_location)

    # save_location = f"../../../../Analysis-Output/Behavioural/Spatial-Density-Plots/dqn_gamma-1/"
    #
    # create_j_turn_2_overlap_plot("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    # create_cstart_overlap_plot("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    # create_routine_turn_overlap_plot("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    #
    # save_location = f"../../../../Analysis-Output/Behavioural/Spatial-Density-Plots/dqn_gamma-2/"
    #
    # create_j_turn_overlap_plot("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    # create_j_turn_2_overlap_plot("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    # create_cstart_overlap_plot("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    # create_routine_turn_overlap_plot("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    #
    # save_location = f"../../../../Analysis-Output/Behavioural/Spatial-Density-Plots/dqn_gamma-4/"
    #
    # create_j_turn_overlap_plot("dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    # create_j_turn_2_overlap_plot("dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    # create_cstart_overlap_plot("dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)
    # create_routine_turn_overlap_plot("dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100, False, save_location)

    # get_all_density_plots_all_subsets(f"dqn_gamma-5", "Behavioural-Data-Free", "Naturalistic", 100, return_objects=False)

    # VERSION 2

    # Getting for combination of models
    # get_all_density_plots_multiple_models(f"dqn_scaffold_14", "Behavioural-Data-Free", "Naturalistic", 10, 4, steps_prior=0,)

    # Getting for individual models across scaffold points
    # plot_all_density_plots_across_scaffold(f"dqn_scaffold_26-1", "Behavioural-Data-Free-A", "Behavioural-Data-Free-B",
    #                                        "Naturalistic", 40)
    # plot_all_density_plots_across_scaffold(f"dqn_scaffold_26-1", "Behavioural-Data-Free-B", "Behavioural-Data-Free-C",
    #                                        "Naturalistic", 40)
    #
    # plot_all_density_plots_across_scaffold(f"dqn_scaffold_26-2", "Behavioural-Data-Free-A", "Behavioural-Data-Free-B",
    #                                        "Naturalistic", 40)
    # plot_all_density_plots_across_scaffold(f"dqn_scaffold_26-2", "Behavioural-Data-Free-B", "Behavioural-Data-Free-C",
    #                                        "Naturalistic", 40)

    # Getting for individual models
    # create_cstart_overlap_plot(f"dqn_predator-1", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False)

    # get_all_density_plots_all_subsets(f"dqn_predator-2", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False)

    # get_all_density_plots_all_subsets(f"dqn_scaffold_26-1", "Behavioural-Data-Free-B", "Naturalistic", 20, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_scaffold_26-1", "Behavioural-Data-Free-C", "Naturalistic", 20, return_objects=False)
    #
    # get_all_density_plots_all_subsets(f"dqn_scaffold_26-2", "Behavioural-Data-Free-A", "Naturalistic", 20, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_scaffold_26-2", "Behavioural-Data-Free-B", "Naturalistic", 20, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_scaffold_26-2", "Behavioural-Data-Free-C", "Naturalistic", 20, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_scaffold_21-2", "Behavioural-Data-Free", "Naturalistic", 40, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_scaffold_22-1", "Behavioural-Data-Free", "Naturalistic", 40, return_objects=False)

    # create_routine_turn_overlap_plot("dqn_scaffold_21-2", "Behavioural-Data-Free", "Naturalistic", 40, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False)
    # get_all_density_plots_all_subsets("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False)

    # create_routine_turn_overlap_plot("dqn_scaffold_26-1", "Behavioural-Data-Free-A", "Naturalistic", 20, return_objects=False)

    # create_cstart_overlap_plot("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False)
    # get_all_density_plots_all_subsets(f"dqn_scaffold_14-2", "Behavioural-Data-Free", "Naturalistic", 10, return_objects=False)
    # for i in range(1, 2):
    #     # get_all_density_plots_all_subsets(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False)
    #     get_all_density_plots_all_subsets(f"dqn_scaffold_18-{i}", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False)
    #
    # get_all_density_plots_all_subsets(f"dqn_scaffold_20-2", "Behavioural-Data-Free", "Naturalistic", 40, return_objects=False)

    # VERSION 1
    # get_all_density_plots_all_subsets("new_even_prey_ref-4", "Behavioural-Data-Free", "Prey", 10)
    # get_all_density_plots_all_subsets("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)

    # get_all_density_plots_all_subsets("new_even_prey_ref-4", "Ablation-Test-Predator_Only-behavioural_data", "Random-Control", 12)
    # get_all_density_plots_all_subsets("new_even_prey_ref-4", "Ablation-Test-Prey-Large-Central-even_naturalistic", "Random-Control", 12)
    # get_all_density_plots_all_subsets("new_even_prey_ref-4", "Ablation-Test-Predator_Only-behavioural_data", "Random-Control", 12)

    # get_all_density_plots_all_subsets("new_even_prey_ref-8", "Behavioural-Data-Free", "Naturalistic", 10)

    # get_all_density_plots_all_subsets("even_prey_ref-7", "Behavioural-Data-Free", "Prey", 10)
    # get_all_density_plots_all_subsets("even_prey_ref-7", "Ablation-Test-Spatial-Density", "Prey-Only-Ablated-100", 3)
    # create_j_turn_overlap_plot("even_prey_ref-7", "Behavioural-Data-Free", "Prey", 10)
    # create_j_turn_overlap_plot("even_prey_ref-7", "Ablation-Test-Spatial-Density", "Prey-Only-Ablated-100", 3)

    # THESE ONES:
    # get_all_density_plots_all_subsets("new_even_prey_ref-8", "Behavioural-Data-Free", "Predator", 10)
    # create_cstart_overlap_plot("even_prey_ref-4", "Behavioural-Data-Free", "Predator", 40)

    # create_routine_turn_overlap_plot("even_prey_ref-5", "Behavioural-Data-Free", "Prey", 10)
    # get_all_density_plots_all_subsets("even_prey_ref-5", "Behavioural-Data-Free", "Prey", 10)

    # get_all_density_plots_all_subsets("even_prey_ref-5", "Behavioural-Data-Free", "Predator", 10)
    # get_all_density_plots_all_subsets("even_prey_ref-6", "Behavioural-Data-Free", "Predator", 10)
    # get_all_density_plots_all_subsets("even_prey_ref-7", "Behavioural-Data-Free", "Predator", 10)

    # get_all_density_plots_all_subsets("new_even_prey_ref-6", "Behavioural-Data-Free", "Prey", 10)
    # get_all_density_plots_all_subsets("new_even_prey_ref-6", "Behavioural-Data-Free", "Naturalistic", 10)
    # get_all_density_plots_all_subsets("new_even_prey_ref-6", "Behavioural-Data-Free", "Predator", 10)
    # # get_all_density_plots_all_subsets("even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)

    # create_cstart_overlap_plot("even_prey_ref-7", "Behavioural-Data-Free", "Predator", 10)
    # create_j_turn_overlap_plot("even_prey_ref-7", "Behavioural-Data-Free", "Prey", 10)
