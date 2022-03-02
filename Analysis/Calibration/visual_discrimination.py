import numpy as np
import matplotlib.pyplot as plt


def load_data(bkg_scatter=0.0005, luminance=[1.0, 0.33]):
    with open(
            f"LuminanceCalibration2/stimulus_present-L{luminance[0]}-BK{bkg_scatter}.npy",
            "rb") as f:
        stimulus_present_1 = np.load(f)
    with open(
            f"LuminanceCalibration2/stimulus_present-L{luminance[1]}-BK{bkg_scatter}.npy",
            "rb") as f:
        stimulus_present_2 = np.load(f)
    with open(
            f"LuminanceCalibration2/stimulus_absent-L{luminance[0]}-BK{bkg_scatter}.npy",
            "rb") as f:
        stimulus_absent_1 = np.load(f)

    with open(
            f"LuminanceCalibration2/stimulus_absent-L{luminance[1]}-BK{bkg_scatter}.npy",
            "rb") as f:
        stimulus_absent_2 = np.load(f)

    with open(f"LuminanceCalibration2/distances_stimulus_present_1-L{luminance[0]}-BK{bkg_scatter}.npy",
              "rb") as f:
        distances_stimulus_present_1 = np.load(f)

    with open(f"LuminanceCalibration2/distances_stimulus_present_2-L{luminance[0]}-BK{bkg_scatter}.npy",
              "rb") as f:
        distances_stimulus_present_2 = np.load(f)

    with open(f"LuminanceCalibration2/distances_stimulus_absent_1-L{luminance[0]}-BK{bkg_scatter}.npy",
              "rb") as f:
        distances_stimulus_absent_1 = np.load(f)

    with open(f"LuminanceCalibration2/distances_stimulus_absent_2-L{luminance[0]}-BK{bkg_scatter}.npy",
              "rb") as f:
        distances_stimulus_absent_2 = np.load(f)

    return distances_stimulus_present_1, distances_stimulus_present_2, distances_stimulus_absent_1, distances_stimulus_absent_2, stimulus_present_1, stimulus_present_2, stimulus_absent_1, stimulus_absent_2


def plot_magnitudes(distances_stimulus_present_1, distances_stimulus_present_2, distances_stimulus_absent_1,
                    distances_stimulus_absent_2, stimulus_present_1, stimulus_present_2, stimulus_absent_1,
                    stimulus_absent_2):

    # Stimulus Present
    z = np.polyfit(distances_stimulus_present_1, stimulus_present_1, 2)
    p = np.poly1d(z)
    ranges = range(int(min(distances_stimulus_present_1)), int(max(distances_stimulus_present_1)))
    plt.plot([r for r in ranges], p([r for r in ranges]), color="y")

    # Stimulus Absent
    z = np.polyfit(distances_stimulus_absent_1, stimulus_absent_1, 2)
    p = np.poly1d(z)
    ranges = range(int(min(distances_stimulus_absent_1)), int(max(distances_stimulus_absent_1)))
    plt.plot([r for r in ranges], p([r for r in ranges]), color="orange")

    plt.scatter(distances_stimulus_absent_1, stimulus_absent_1, alpha=0.005)
    plt.scatter(distances_stimulus_present_1, stimulus_present_1, color="r")
    plt.show()

    z = np.polyfit(distances_stimulus_present_2, stimulus_present_2, 2)
    p = np.poly1d(z)
    ranges = range(int(min(distances_stimulus_present_2)), int(max(distances_stimulus_present_2)))
    plt.plot([r for r in ranges], p([r for r in ranges]), color="y")

    # Stimulus Absent
    z = np.polyfit(distances_stimulus_absent_2, stimulus_absent_2, 2)
    p = np.poly1d(z)
    ranges = range(int(min(distances_stimulus_absent_2)), int(max(distances_stimulus_absent_2)))
    plt.plot([r for r in ranges], p([r for r in ranges]), color="orange")

    plt.scatter(distances_stimulus_absent_2, stimulus_absent_2, alpha=0.005)
    plt.scatter(distances_stimulus_present_2, stimulus_present_2, color="r")
    plt.show()


def get_max_values_only(distances_stimulus_present_1, distances_stimulus_present_2, stimulus_present_1, stimulus_present_2):
    all_distances_1 = np.unique(distances_stimulus_present_1)
    all_distances_2 = np.unique(distances_stimulus_present_2)

    new_stimulus_present_1 = []
    new_stimulus_present_2 = []

    for d in all_distances_1:
        new_stimulus_present_1.append(max(stimulus_present_1[distances_stimulus_present_1 == d]))

    for d in all_distances_2:
        new_stimulus_present_2.append(max(stimulus_present_2[distances_stimulus_present_2 == d]))

    return all_distances_1, all_distances_2, np.array(new_stimulus_present_1), np.array(new_stimulus_present_2)


def get_percentage_discriminable(distances_absent, distances_present, absent_points, present_points):
    all_distances = np.unique(distances_absent)

    all_percentages = []

    for d in all_distances:
        max_absent = max(absent_points[distances_absent == d])
        points = present_points[distances_present == d]
        points_above = np.sum((points > max_absent)*1)
        points_below = np.sum((points <= max_absent)*1)

        percentage_above = (points_above/(points_below+points_above)) * 100
        all_percentages.append(percentage_above)

    z = np.polyfit(distances_present, present_points, 2)
    p = np.poly1d(z)
    ranges = range(int(min(distances_present)), int(max(distances_present)))
    plt.plot([r for r in ranges], p([r for r in ranges]), color="orange")

    plt.scatter(all_distances, all_percentages)
    plt.show()


distances_stimulus_present_1, distances_stimulus_present_2, distances_stimulus_absent_1, distances_stimulus_absent_2, \
stimulus_present_1, stimulus_present_2, stimulus_absent_1, stimulus_absent_2 = load_data(0.0005, [1.0, 0.5])

plot_magnitudes(distances_stimulus_present_1, distances_stimulus_present_2, distances_stimulus_absent_1,
                distances_stimulus_absent_2, stimulus_present_1, stimulus_present_2, stimulus_absent_1,
                stimulus_absent_2)

distances_stimulus_present_1, distances_stimulus_present_2, stimulus_present_1, stimulus_present_2 = get_max_values_only(distances_stimulus_present_1, distances_stimulus_present_2, stimulus_present_1, stimulus_present_2)

plot_magnitudes(distances_stimulus_present_1, distances_stimulus_present_2, distances_stimulus_absent_1,
                distances_stimulus_absent_2, stimulus_present_1, stimulus_present_2, stimulus_absent_1,
                stimulus_absent_2)

# Discriminable plot - whether points fall above the top line of blue

get_percentage_discriminable(distances_stimulus_absent_1, distances_stimulus_present_1, stimulus_absent_1, stimulus_present_1)

get_percentage_discriminable(distances_stimulus_absent_2, distances_stimulus_present_2, stimulus_absent_2, stimulus_present_2)

# Construct metric - percentage discriminable.
