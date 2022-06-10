import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_action_name import get_action_name
from Analysis.Behavioural.Tools.anchored_scale_bar import AnchoredHScaleBar

"""
To create a graph of the style in Figure 3b of Marques et al. (2018)
"""


def draw_fish(x, y, mouth_size, head_size, tail_length, ax):
    mouth_centre = (x, y)  # TODO: Make sure this is true

    mouth = plt.Circle(mouth_centre, mouth_size,  fc="green")
    ax.add_patch(mouth)

    angle = (2 * np.pi)/2
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
        v = [i[0]-fish_position[0], i[1]-fish_position[1]]
        theta = 2 * np.pi - fish_orientation
        tran_v = [np.cos(theta) * v[0]-np.sin(theta) * v[1],
                  np.sin(theta) * v[0]+np.cos(theta) * v[1]]
        tran_v[0] = tran_v[0] + 300
        tran_v[1] = tran_v[1] + 300
        transformed_coordinates.append(tran_v)
    return transformed_coordinates


def get_clouds_with_action(data, action=0):
    prey_cloud = []
    predator_cloud = []

    for i, step in enumerate(data["action"]):
        if data["action"][i] == action:
            allocentric_prey, allocentric_predators = get_nearby_features(data, i)

            if len(allocentric_prey) > 0:
                egocentric_prey = transform_to_egocentric(allocentric_prey, data["fish_position"][i], data["fish_angle"][i])
            else:
                egocentric_prey = []

            if len(allocentric_predators) > 0:
                egocentric_predators = transform_to_egocentric(allocentric_predators, data["fish_position"][i], data["fish_angle"][i])
            else:
                egocentric_predators = []

            prey_cloud = prey_cloud + egocentric_prey
            predator_cloud = predator_cloud + egocentric_predators

    return prey_cloud, predator_cloud


def create_density_cloud(density_list, action_num, stimulus_name, return_objects):
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
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="Reds", )#cmap='gist_gray')#  cmap='PuBu_r')

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)
    # plt.arrow(-300, 220, 0, 40, width=10, color="red")

    ax = draw_fish(-300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(-600, 0)
    ax.set_ylim(-80, 520)

    # plt.scatter(y, x, c="b", s=1)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: {stimulus_name}, Action: {get_action_name(action_num)}, N-Samples: {n_samples}")

    if return_objects:
        plt.clf()
        return ax
    else:
        plt.show()


def get_all_density_plots(data):
    for action_num in range(0, 10):
        prey_1, pred_1 = get_clouds_with_action(data, action_num)

        if len(prey_1) > 2:
            create_density_cloud(prey_1, action_num, "Prey")

        if len(pred_1) > 2:
            create_density_cloud(pred_1, action_num, "Predator")


def get_all_density_plots_all_subsets(p1, p2, p3, n, return_objects):
    axes_objects = []
    for action_num in range(0, 10):
        prey_cloud = []
        pred_cloud = []
        for i in range(1, n+1):
            if i > 100:
                data = load_data(p1, f"{p2}-2", f"{p3} {i}")
            else:
                data = load_data(p1, p2, f"{p3}-{i}")

            prey_1, pred_1 = get_clouds_with_action(data, action_num)
            prey_cloud = prey_cloud + prey_1
            pred_cloud = pred_cloud + pred_1

        if len(prey_cloud) > 2:
            ax = create_density_cloud(prey_cloud, action_num, "Prey", return_objects)
            axes_objects.append(ax)

        if len(pred_cloud) > 2:
            ax = create_density_cloud(pred_cloud, action_num, "Predator", return_objects)
            axes_objects.append(ax)

    ax = create_j_turn_overlap_plot(p1, p2, p3, n, return_objects)
    axes_objects.append(ax)
    ax = create_routine_turn_overlap_plot(p1, p2, p3, n, return_objects)
    axes_objects.append(ax)
    ax = create_cstart_overlap_plot(p1, p2, p3, n, return_objects)
    axes_objects.append(ax)
    return axes_objects


def create_j_turn_overlap_plot(p1, p2, p3, n, return_objects):
    prey_cloud_left = []
    for i in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 4)
        prey_cloud_left = prey_cloud_left + prey_1
    prey_cloud_right = []
    for i in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 5)
        prey_cloud_right = prey_cloud_right + prey_1
    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    #y = np.negative(y)
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
    #y = np.negative(y)
    nbins = 300
    try:
        k = kde.gaussian_kde([y, x])
    except ValueError:
        return
    zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi - zi2

    # Make the plot
    fig, ax = plt.subplots()

    pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape),  cmap='RdBu')

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)

    ax = draw_fish(300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(0, 600)
    ax.set_ylim(-80, 520)

    fig.colorbar(pcm)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Prey, Action: J-turns, N-Samples: {n_samples}")

    if return_objects:
        plt.clf()
        return ax
    else:
        plt.show()


def create_routine_turn_overlap_plot(p1, p2, p3, n, return_objects):
    prey_cloud_left = []
    for i in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 1)
        prey_cloud_left = prey_cloud_left + prey_1
    prey_cloud_right = []
    for i in range(1, n+1):
        data = load_data(p1, p2, f"{p3}-{i}")
        prey_1, pred_1 = get_clouds_with_action(data, 2)
        prey_cloud_right = prey_cloud_right + prey_1
    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    #y = np.negative(y)
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
    #y = np.negative(y)
    nbins = 300
    try:
        k = kde.gaussian_kde([y, x])
    except ValueError:
        return
    zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))

    zi = zi2 - zi

    fig, ax = plt.subplots()

    pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdBu')

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)

    ax = draw_fish(-300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(-600, 0)
    ax.set_ylim(-80, 520)

    fig.colorbar(pcm)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Prey, Action: Routine turns, N-Samples: {n_samples}")

    if return_objects:
        plt.clf()
        return ax
    else:
        plt.show()


def create_cstart_overlap_plot(p1, p2, p3, n, return_objects):
    prey_cloud_left = []
    for i in range(1, n+1):
        if i < 11: continue
            # print(i)
            #
            # data = load_data(p1, f"{p2}-2", f"{p3}-{i}")
        else:
            print(i)
            data = load_data(p1, p2, f"{p3} {i}")
        prey_1, pred_1 = get_clouds_with_action(data, 7)
        prey_cloud_left = prey_cloud_left + prey_1
    prey_cloud_right = []
    for i in range(1, n+1):
        if i < 11: continue
            # data = load_data(p1, f"{p2}-2", f"{p3} {i}")
        else:
            data = load_data(p1, p2, f"{p3} {i}")
        prey_1, pred_1 = get_clouds_with_action(data, 8)
        prey_cloud_right = prey_cloud_right + prey_1
    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    #y = np.negative(y)
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
    #y = np.negative(y)
    nbins = 300
    try:
        k = kde.gaussian_kde([y, x])
    except ValueError:
        return
    zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))

    zi = zi - zi2
    # Make the plot
    fig, ax = plt.subplots()

    pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='RdBu')

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)

    ax = draw_fish(300, 220, 4, 2.5, 41.5, ax)
    ax.set_xlim(0, 600)
    ax.set_ylim(-80, 520)

    fig.colorbar(pcm)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Predator, Action: C-Starts, N-Samples: {n_samples}")

    if return_objects:
        plt.clf()
        return ax
    else:
        plt.show()


def create_j_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2):
    prey_cloud_left = []
    prey_cloud_right = []
    for m in range(1, n2+1):
        for i in range(1, n+1):
            data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")
            prey_1, pred_1 = get_clouds_with_action(data, 4)
            prey_cloud_left = prey_cloud_left + prey_1
        for i in range(1, n+1):
            data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")
            prey_1, pred_1 = get_clouds_with_action(data, 5)
            prey_cloud_right = prey_cloud_right + prey_1
    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    #y = np.negative(y)
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
    #y = np.negative(y)
    nbins = 300
    k = kde.gaussian_kde([y, x])
    zi2 = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi - zi2

    # Make the plot
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape),  cmap='RdBu')

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

    for m in range(1, n2+1):
        for i in range(1, n+1):
            data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")
            prey_1, pred_1 = get_clouds_with_action(data, 1)
            prey_cloud_left = prey_cloud_left + prey_1
        for i in range(1, n+1):
            data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")
            prey_1, pred_1 = get_clouds_with_action(data, 2)
            prey_cloud_right = prey_cloud_right + prey_1

    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    #y = np.negative(y)
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # For right
    x = np.array([i[0] for i in prey_cloud_right])
    y = np.array([i[1] for i in prey_cloud_right])
    #y = np.negative(y)
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
    for m in range(1, n2+1):

        for i in range(1, n+1):
            if i < 11: continue
                # print(i)
                #
                # data = load_data(p1, f"{p2}-2", f"{p3}-{i}")
            else:
                print(i)
                data = load_data(f"{p1}-{m}", p2, f"{p3} {i}")
            prey_1, pred_1 = get_clouds_with_action(data, 7)
            prey_cloud_left = prey_cloud_left + prey_1
        for i in range(1, n+1):
            if i < 11: continue
                # data = load_data(p1, f"{p2}-2", f"{p3} {i}")
            else:
                data = load_data(f"{p1}-{m}", p2, f"{p3} {i}")
            prey_1, pred_1 = get_clouds_with_action(data, 8)
            prey_cloud_right = prey_cloud_right + prey_1
    n_samples = len(prey_cloud_left) + len(prey_cloud_right)
    # For left
    x = np.array([i[0] for i in prey_cloud_left])
    y = np.array([i[1] for i in prey_cloud_left])
    #y = np.negative(y)
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # For right
    x = np.array([i[0] for i in prey_cloud_right])
    y = np.array([i[1] for i in prey_cloud_right])
    #y = np.negative(y)
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


def get_all_density_plots_multiple_models(p1, p2, p3, n, n2):
    for action_num in range(0, 10):
        prey_cloud = []
        pred_cloud = []
        for m in range(1, n2+1):
            for i in range(1, n+1):
                if i > 12:
                    data = load_data(f"{p1}-{m}", f"{p2}-2", f"{p3} {i}")
                else:
                    data = load_data(f"{p1}-{m}", p2, f"{p3}-{i}")

                prey_1, pred_1 = get_clouds_with_action(data, action_num)
                prey_cloud = prey_cloud + prey_1
                pred_cloud = pred_cloud + pred_1

        if len(prey_cloud) > 2:
            create_density_cloud(prey_cloud, action_num, "Prey")

        if len(pred_cloud) > 2:
            create_density_cloud(pred_cloud, action_num, "Predator")

    create_j_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2)
    create_routine_turn_overlap_plot_multiple_models(p1, p2, p3, n, n2)
    create_cstart_overlap_plot_multiple_models(p1, p2, p3, n, n2)


# VERSION 2

# Getting for combination of models
# get_all_density_plots_multiple_models(f"dqn_scaffold_14", "Behavioural-Data-Free", "Naturalistic", 10, 4)

# Getting for individual models
for i in range(1, 2):
    get_all_density_plots_all_subsets(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False)

# for i in range(1, 3):
#     get_all_density_plots_all_subsets(f"dqn_scaffold_15-{i}", "Behavioural-Data-Free", "Naturalistic", 10)

# for i in range(1, 3):
#     get_all_density_plots_all_subsets(f"dqn_scaffold_16-{i}", "Behavioural-Data-Free", "Naturalistic", 10)
#
# for i in range(1, 3):
#     get_all_density_plots_all_subsets(f"dqn_scaffold_17-{i}", "Behavioural-Data-Free", "Naturalistic", 10)
#
# for i in range(1, 3):
#     get_all_density_plots_all_subsets(f"dqn_scaffold_18-{i}", "Behavioural-Data-Free", "Naturalistic", 10)


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

#THESE ONES:
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



