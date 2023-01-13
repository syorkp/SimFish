import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_salt_data import get_salt_data
from Analysis.Behavioural.Tools.anchored_scale_bar import AnchoredHScaleBar
from Analysis.Behavioural.Tools.random_data_generators import generate_random_fish_position_data


#                TOOLS

def get_fish_salt_source_incidence(fish_positions, fish_orientation, salt_source_locations):
    fish_positions = np.array(fish_positions)
    fish_orientation = np.array(fish_orientation)
    salt_source_locations = np.array(salt_source_locations)

    fish_orientation_sign = ((fish_orientation >= 0) * 1) + ((fish_orientation < 0) * -1)

    # Remove full orientations (so is between -2pi and 2pi
    fish_orientation %= 2 * np.pi * fish_orientation_sign

    # Convert to positive scale between 0 and 2pi
    fish_orientation[fish_orientation < 0] += 2 * np.pi

    fish_salt_vectors = salt_source_locations - fish_positions

    # Adjust according to quadrents.
    fish_salt_angles = np.arctan(fish_salt_vectors[:, :, 1] / fish_salt_vectors[:, :, 0])

    #   Generates positive angle from left x axis clockwise.
    # UL quadrent
    in_ul_quadrent = (fish_salt_vectors[:, :, 0] < 0) * (fish_salt_vectors[:, :, 1] > 0)
    fish_salt_angles[in_ul_quadrent] += np.pi
    # BR quadrent
    in_br_quadrent = (fish_salt_vectors[:, :, 0] > 0) * (fish_salt_vectors[:, :, 1] < 0)
    fish_salt_angles[in_br_quadrent] += (np.pi * 2)
    # BL quadrent
    in_bl_quadrent = (fish_salt_vectors[:, :, 0] < 0) * (fish_salt_vectors[:, :, 1] < 0)
    fish_salt_angles[in_bl_quadrent] += np.pi

    # Angle ends up being between 0 and 2pi as clockwise from right x-axis. Same frame as fish angle:
    fish_salt_incidence = fish_orientation - fish_salt_angles

    fish_salt_incidence[fish_salt_incidence > np.pi] %= np.pi
    fish_salt_incidence[fish_salt_incidence < -np.pi] %= -np.pi

    return fish_salt_incidence



#                PLOTS

def display_2d_kdf_salt_fish_position(fish_positions, salt_locations):
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    salt_locations_flattened = np.concatenate(salt_locations, axis=0)

    relative_positions = salt_locations_flattened - fish_positions_flattened

    x = np.array([i[0] for i in list(relative_positions)])
    y = np.array([i[1] for i in list(relative_positions)])
    # y = np.negative(y)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([y, x])
    yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    fig, ax = plt.subplots()

    ax.pcolormesh(xi, yi, zi.reshape(xi.shape))

    plt.scatter(0, 0, color="red")
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.title(f"Fish-Salt Relative Positions")

    plt.savefig("../../../Analysis-Output/Behavioural/Salt/fish_salt_relative_positions.jpg")
    plt.clf()
    plt.close()


def plot_fish_salt_distance_density(fish_positions, salt_locations, w=1500, h=1500):
    d_max = (w ** 2 + h ** 2) ** 0.5
    fish_positions_flattened = np.concatenate(fish_positions, axis=0)
    salt_locations_flattened = np.concatenate(salt_locations, axis=0)

    relative_positions = salt_locations_flattened - fish_positions_flattened
    distances = (relative_positions[:, 0] ** 2 + relative_positions[:, 1] ** 2) ** 0.5

    random_fish_positions = generate_random_fish_position_data(n=fish_positions_flattened.shape[0], w=w, h=h)
    relative_positions = salt_locations_flattened - random_fish_positions
    distances_simulated = (relative_positions[:, 0] ** 2 + relative_positions[:, 1] ** 2) ** 0.5

    plt.title("Histogram of Simulated Fish Position-Salt Source Distance")
    plt.hist(distances, bins=100, range=[0, d_max], alpha=0.5, color="b", label="Fish Distances")
    plt.hist(distances_simulated, bins=100, range=[0, d_max], alpha=0.5, color="r", label="Simulated Fish Distances")
    plt.xlabel("Distance From Salt Source")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig("../../../Analysis-Output/Behavioural/Salt/histogram_of_salt_concentration_fish_density.jpg")
    plt.clf()
    plt.close()


def plot_salt_concentration_against_turn_size_scatter(fish_orientations, fish_positions, salt_locations, salt_concentrations):
    fish_salt_incidence = get_fish_salt_source_incidence(fish_positions, fish_orientations, salt_locations)
    fish_salt_incidence = np.absolute(fish_salt_incidence)
    fish_salt_incidence_change = fish_salt_incidence[:, 1:] - fish_salt_incidence[:, :-1]
    turns_away = (fish_salt_incidence_change < 0)
    turns_towards = (fish_salt_incidence_change >= 0)

    salt_concentrations = np.array(salt_concentrations)
    relevant_salt_concentrations = salt_concentrations[:, :-1]

    fish_orientations = np.array(fish_orientations)
    fish_turns = fish_orientations[:, 1:] - fish_orientations[:, :-1]

    plt.scatter(fish_turns[turns_away].flatten(), relevant_salt_concentrations[turns_away].flatten(), color="b")
    plt.scatter(fish_turns[turns_towards].flatten(), relevant_salt_concentrations[turns_towards].flatten(), color="r")

    plt.savefig("../../../Analysis-Output/Behavioural/Salt/fish_turns_salt_gradient.jpg")
    plt.clf()
    plt.close()


def plot_salt_concentration_against_turn_laterality_hist(fish_orientations, fish_positions, salt_locations, salt_concentrations):
    fish_salt_incidence = get_fish_salt_source_incidence(fish_positions, fish_orientations, salt_locations)
    fish_salt_incidence = np.absolute(fish_salt_incidence)
    fish_salt_incidence_change = fish_salt_incidence[:, 1:] - fish_salt_incidence[:, :-1]
    turns_away = (fish_salt_incidence_change < 0)
    turns_towards = (fish_salt_incidence_change >= 0)

    print(np.sum(turns_away * 1))
    print(np.sum(turns_towards * 1))

    salt_concentrations = np.array(salt_concentrations)
    relevant_salt_concentrations = salt_concentrations[:, :-1]

    fish_orientations = np.array(fish_orientations)
    fish_turns = fish_orientations[:, 1:] - fish_orientations[:, :-1]

    # plt.hist(fish_turns[turns_away].flatten(), relevant_salt_concentrations[turns_away].flatten(), color="b")
    # plt.hist(fish_turns[turns_towards].flatten(), relevant_salt_concentrations[turns_towards].flatten(), color="r")
    plt.hist([relevant_salt_concentrations[turns_away].flatten(), relevant_salt_concentrations[turns_towards].flatten()], bins=20)#, color="b")
    # plt.hist(relevant_salt_concentrations[turns_towards].flatten(), bins=100, color="r")

    plt.savefig("../../../Analysis-Output/Behavioural/Salt/fish_turns_salt_gradient_hist.jpg")
    plt.clf()
    plt.close()



if __name__ == "__main__":
    compiled_fish_positions = []
    compiled_fish_orientations = []
    compiled_salt_source_locations = []
    compiled_salt_concentrations = []
    for i in range(2, 3):
        fish_positions, fish_orientations, salt_source_locations, salt_concentrations = \
            get_salt_data(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10)
        # display_2d_kdf_salt_fish_position(fish_positions, salt_source_locations)
        # plot_fish_salt_distance_density(fish_positions, salt_source_locations)
        plot_salt_concentration_against_turn_laterality_hist(fish_orientations, fish_positions, salt_source_locations, salt_concentrations)
        # plot_salt_concentration_against_turn_direction(fish_orientations, fish_positions,
        #                                                salt_source_locations, salt_concentrations)
        compiled_fish_positions += fish_positions
        compiled_fish_orientations += fish_orientations
        compiled_salt_source_locations += salt_source_locations
        compiled_salt_concentrations += salt_concentrations

    # Salt kdf.
    # display_2d_kdf_salt_fish_position(compiled_fish_positions, compiled_salt_source_locations)
    # plot_fish_salt_distance_density(compiled_fish_positions, compiled_salt_source_locations)

    # Salt-direction plot
# /
    # plot_salt_concentration_against_turn_laterality(compiled_fish_orientations, compiled_fish_positions,
    #                                                 compiled_salt_source_locations, compiled_salt_concentrations)


