import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

from Analysis.load_data import load_data

"""
To create a graph of the style in Figure 3b of Marques et al. (2018)
"""


def get_nearby_features(data, step, proximity=300):
    """For a single step, returns the positions of nearby prey and predators."""
    nearby_prey_coordinates = []
    nearby_predator_coordinates = []
    nearby_area = [[data["position"][step][0] - proximity,
                   data["position"][step][0] + proximity],
                   [data["position"][step][1] - proximity,
                    data["position"][step][1] + proximity]
                   ]

    for i in data["prey_positions"][step]:
        is_in_area = nearby_area[0][0] <= i[0] <= nearby_area[0][1] and \
                     nearby_area[1][0] <= i[1] <= nearby_area[1][1]
        if is_in_area:
            nearby_prey_coordinates.append(i)
    j = data["predator_position"][step]
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

    for i, step in enumerate(data["step"]):
        if data["behavioural choice"][i] == action:
            allocentric_prey, allocentric_predators = get_nearby_features(data, i)

            if len(allocentric_prey) > 0:
                egocentric_prey = transform_to_egocentric(allocentric_prey, data["position"][i], data["fish_angle"][i])
            else:
                egocentric_prey = []

            if len(allocentric_predators) > 0:
                egocentric_predators = transform_to_egocentric(allocentric_predators, data["position"][i], data["fish_angle"][i])
            else:
                egocentric_predators = []

            prey_cloud = prey_cloud + egocentric_prey
            predator_cloud = predator_cloud + egocentric_predators

    return prey_cloud, predator_cloud


def get_action_name(action_num):
    if action_num == 0:
        action_name = "Slow2"
    elif action_num == 1:
        action_name = "RT Right"
    elif action_num == 2:
        action_name = "RT Left"
    elif action_num == 3:
        action_name = "sCS"
    elif action_num == 4:
        action_name = "J-turn Right"
    elif action_num == 5:
        action_name = "J-turn Left"
    elif action_num == 6:
        action_name = "Rest"
    elif action_num == 7:
        action_name = "SLC Right"
    elif action_num == 8:
        action_name = "SLC Left"
    elif action_num == 9:
        action_name = "AS"
    else:
        action_name = "None"
    return action_name


from matplotlib.lines import Line2D
import matplotlib
class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None,
                 frameon=True, linekw={}, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **linekw)
        vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
        vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],
                                 align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                 borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                 **kwargs)


def create_density_cloud(density_list, action_num, stimulus_name):
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

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape))

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)
    plt.arrow(-300, 220, 0, 40, width=10, color="red")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: {stimulus_name}, Action: {get_action_name(action_num)}")
    plt.show()


def get_all_density_plots(data):
    for action_num in range(0, 10):
        prey_1, pred_1 = get_clouds_with_action(data, action_num)

        if len(prey_1) > 2:
            create_density_cloud(prey_1, action_num, "Prey")

        if len(pred_1) > 2:
            create_density_cloud(pred_1, action_num, "Predator")


def get_all_density_plots_all_subsets(p1, p2, p3, n):
    for action_num in range(0, 10):
        prey_cloud = []
        pred_cloud = []
        for i in range(1, n+1):
            if i > 12:
                data = load_data(p1, f"{p2}-2", f"{p3} {i}")
            else:
                data = load_data(p1, p2, f"{p3}-{i}")

            prey_1, pred_1 = get_clouds_with_action(data, action_num)
            prey_cloud = prey_cloud + prey_1
            pred_cloud = pred_cloud + pred_1

        if len(prey_cloud) > 2:
            create_density_cloud(prey_cloud, action_num, "Prey")

        if len(pred_cloud) > 2:
            create_density_cloud(pred_cloud, action_num, "Predator")


def create_j_turn_overlap_plot(p1, p2, p3, n):
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

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape),  cmap='RdBu')

    ob = AnchoredHScaleBar(size=100, label="10mm", loc=4, frameon=True,
                           pad=0.6, sep=4, linekw=dict(color="crimson"), )
    ax.add_artist(ob)
    plt.arrow(300, 220, 0, 40, width=10, color="red")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Prey, Action: J-turns")
    plt.show()




def create_routine_turn_overlap_plot(p1, p2, p3, n):
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
    plt.arrow(300, 220, 0, 40, width=10, color="red")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Prey, Action: Routine turns")
    plt.show()


def create_cstart_overlap_plot(p1, p2, p3, n):
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
    plt.arrow(300, 220, 0, 40, width=10, color="red")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Feature: Predator, Action: C-Starts")
    plt.show()

#
# get_all_density_plots_all_subsets("new_even_prey_ref-4", "Behavioural-Data-Free", "Prey", 10)
# get_all_density_plots_all_subsets("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)

#
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

# get_all_density_plots_all_subsets("even_prey_ref-5", "Behavioural-Data-Free", "Predator", 10)
# get_all_density_plots_all_subsets("even_prey_ref-6", "Behavioural-Data-Free", "Predator", 10)
# get_all_density_plots_all_subsets("even_prey_ref-7", "Behavioural-Data-Free", "Predator", 10)
#
#
#
# get_all_density_plots_all_subsets("new_even_prey_ref-6", "Behavioural-Data-Free", "Prey", 10)
# get_all_density_plots_all_subsets("new_even_prey_ref-6", "Behavioural-Data-Free", "Naturalistic", 10)
# get_all_density_plots_all_subsets("new_even_prey_ref-6", "Behavioural-Data-Free", "Predator", 10)
#
# # get_all_density_plots_all_subsets("even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)

# create_cstart_overlap_plot("even_prey_ref-7", "Behavioural-Data-Free", "Predator", 10)
# create_j_turn_overlap_plot("even_prey_ref-7", "Behavioural-Data-Free", "Prey", 10)



