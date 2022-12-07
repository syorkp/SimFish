import math

import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression


# spot_data1 = np.genfromtxt("Prey-Tracking/spots_0-5000.csv", delimiter=',')
# track_data1 = np.genfromtxt("Prey-Tracking/tracks_0-5000.csv", delimiter=',')

# spot_data1 = np.genfromtxt("Prey-Tracking/spots_1.csv", delimiter=',')
# track_data1 = np.genfromtxt("Prey-Tracking/tracks_1.csv", delimiter=',')
# track_data1 = np.genfromtxt("Prey-Tracking/fish_track.csv", delimiter=',')

# track_counts, tracks = np.unique(track_data1, return_counts=True)
#
#
# # track_id_spots = spot_data1[4:, 2]
# track_id_tracks = track_data1[4:, 2]
# track_positions = track_data1[4:, 4:6]
# track_timestamps = track_data1[4:, 7]

# track_id_tracks = track_id_tracks[track_timestamps < 5000]
# track_positions_1 = track_positions[track_id_tracks == 0]

# track_counts, tracks = np.unique(track_id_tracks, return_counts=True)
# x = {str(t): count for t, count in zip(tracks, track_counts)}


# plt.hist(track_id_tracks, 100)
# plt.show()


# for i in range(80, 100):
#     track_positions_1 = track_positions[track_id_tracks == i]
#
#     plt.scatter(track_positions_1[:, 0], track_positions_1[:, 1])
#     plt.title(f"{i}")
#     plt.show()


# FISH TRACK

def load_fish_data():
    fish_track_data = np.genfromtxt("Particle-Tracking-Data/fish_track.csv", delimiter=',')

    # track_id_spots = spot_data1[4:, 2]
    fish_track_id_tracks = fish_track_data[4:, 2]
    fish_track_positions = fish_track_data[4:, 4:6]
    fish_track_timestamps = fish_track_data[4:, 7]

    possible_fish_tracks = [9, 28, 31, 39, 43, 55, 56, 59, 64, 69, 76, 79, 86, 87, 91, 95]

    fish_track_full = np.zeros((0, 2))
    fish_timestamps_full = np.zeros((0))

    for i in possible_fish_tracks:
        fish_track_full = np.concatenate((fish_track_full, fish_track_positions[fish_track_id_tracks == i]), axis=0)
        fish_timestamps_full = np.concatenate((fish_timestamps_full, fish_track_timestamps[fish_track_id_tracks == i]), axis=0)
        # track_positions_1 = track_positions[track_id_tracks == i]
        # track_timestamps_1 = track_timestamps[track_id_tracks == i]
        # print(f"Track: {i}, min_t: {min(track_timestamps_1)}, max_t: {max(track_timestamps_1)}")

    # plt.scatter(fish_track_full[:, 0], fish_track_full[:, 1], c=fish_timestamps_full)
    # plt.show()

    return fish_track_full, fish_timestamps_full


def compute_fish_orientation(fish_track_positions, fish_track_timestamps):
    reordering = fish_track_timestamps.argsort()
    fish_track_positions = fish_track_positions[reordering, :]
    differences = (fish_track_positions[1:, :] - fish_track_positions[:-1, :]) + 0.000001  # Added to prevent zerodiv error
    o_over_a = differences[:, 1]/differences[:, 0]
    orientations = np.arctan(o_over_a)

    UL_quadrant = ((differences[:, 0] < 0) * 1) * ((differences[:, 1] < 0) * 1) * np.pi
    UR_quadrant = ((differences[:, 0] > 0) * 1) * ((differences[:, 1] < 0) * 1) * np.pi * 2
    BL_quadrant = ((differences[:, 0] < 0) * 1) * ((differences[:, 1] > 0) * 1) * np.pi

    orientations += UL_quadrant + UR_quadrant + BL_quadrant

    large_transitions = ((((differences[:, 0] ** 2) + (differences[:, 0] ** 2)) ** 0.5) > 1) * 1
    orientations = orientations * large_transitions

    i = 0
    n = len(orientations) - 1
    while True:
        if orientations[i] == 0:
            orientations[i] = (orientations[i-1])
        else:
            orientations[i] = (orientations[i])
        if i == n:
            break
        else:
            i += 1

    # Put back in previous order
    # orientations = orientations[fish_track_timestamps.astype(int)]

    # if differences[0] < 0 and differences[1] < 0:
    #     # Generates postiive angle from left x axis clockwise.
    #     # print("UL quadrent")
    #     angle += np.pi
    # elif vector[1] < 0:
    #     # Generates negative angle from right x axis anticlockwise.
    #     # print("UR quadrent.")
    #     angle = angle + (np.pi * 2)
    # elif vector[0] < 0:
    #     # Generates negative angle from left x axis anticlockwise.
    #     # print("BL quadrent.")
    #     angle = angle + np.pi

    return orientations, np.array(sorted(fish_track_timestamps[1:])).astype(int)


def get_transformed_fish_positions(fish_positions, fish_orientations, offset=4*math.sqrt(13)):
    # Create desired vector
    x_transform = np.cos(fish_orientations) * offset
    x_transform = np.expand_dims(x_transform, 1)
    y_transform = np.sin(fish_orientations) * offset
    y_transform = np.expand_dims(y_transform, 1)

    transform = np.concatenate((x_transform, y_transform), axis=1)
    fish_positions = fish_positions + transform
    return fish_positions


fish_track_full, fish_timestamps_full = load_fish_data()
orientations, ordered_timestamps = compute_fish_orientation(fish_track_full, fish_timestamps_full)
orientations = np.concatenate((orientations[0:1], orientations), axis=0)

# Order fish position
reordering = fish_timestamps_full.argsort()
fish_track_full = fish_track_full[reordering, :]


# Particle position - 591, 653
# Actual head position - 583, 641
# TO adjust position, need to add 4 * root2(13) in direction to transform coordinate.
transformed_fish_position = get_transformed_fish_positions(fish_track_full, orientations)
ordered_timestamps = np.concatenate((ordered_timestamps[0:1]-1, ordered_timestamps), axis=0)

#                 Finding proximal paramecia


def get_proximal_paramecia(fish_track_full, fish_timestamps_full):
    max_distance = 100
    paramecia_tracks = np.genfromtxt("Particle-Tracking-Data/tracks_1.csv", delimiter=',')
    # limited_paramecia_tracks = np.genfromtxt("Prey-Tracking/tracks_0-5000.csv", delimiter=',')

    paramecia_id_tracks = paramecia_tracks[4:, 2]
    paramecia_positions = paramecia_tracks[4:, 4:6]
    paramecia_timestamps = paramecia_tracks[4:, 7]

    proximal_paramecia_identities = []
    proximal_paramecia_timestamps = []
    proximal_paramecia_positions = []

    for paramecia in range(0, int(max(paramecia_id_tracks))):
        p_positions = paramecia_positions[paramecia_id_tracks == paramecia]
        p_timestamps = paramecia_timestamps[paramecia_id_tracks == paramecia]

        for t in p_timestamps:
            if t in fish_timestamps_full:
                par = p_positions[p_timestamps == t]
                fsh = fish_track_full[fish_timestamps_full == t]
                distance = ((par[0, 0]-fsh[0, 0])**2 + (par[0, 1]-fsh[0, 1])**2) ** 0.5
                if distance < max_distance:
                    proximal_paramecia_timestamps.append(t)
                    proximal_paramecia_identities.append(paramecia)
                    proximal_paramecia_positions.append(par[0])

    return proximal_paramecia_timestamps, proximal_paramecia_identities, proximal_paramecia_positions

# proximal_paramecia_timestamps, proximal_paramecia_identities, proximal_paramecia_positions = get_proximal_paramecia(fish_track_full, fish_timestamps_full)
#
# #                    Saving Interactions
#
# with open('proximal_paramecia_timestamps2.npy', 'wb') as outfile:
#     np.save(outfile, proximal_paramecia_timestamps)
#
# with open('proximal_paramecia_identities2.npy', 'wb') as outfile:
#     np.save(outfile, proximal_paramecia_identities)

# with open('proximal_paramecia_positions.npy', 'wb') as outfile:
#     np.save(outfile, np.array(proximal_paramecia_positions))

# # with open('fish_track_full.npy', 'wb') as outfile:
# #     np.save(outfile, fish_track_full)
#
# # paramecia_tracks = np.genfromtxt("Prey-Tracking/tracks_1.csv", delimiter=',')
# # paramecia_positions = paramecia_tracks[4:, 4:6]


def get_full_paramecia_data():
    paramecia_tracks = np.genfromtxt("Particle-Tracking-Data/tracks_1.csv", delimiter=',')
    paramecia_id_tracks = paramecia_tracks[4:, 2]
    paramecia_positions = paramecia_tracks[4:, 4:6]
    paramecia_timestamps = paramecia_tracks[4:, 7]
    return paramecia_id_tracks, paramecia_positions, paramecia_timestamps


# full_paramecia_id_tracks, full_paramecia_positions, full_paramecia_timestamps = get_full_paramecia_data()
#
#
# with open('Prey-Tracking/full_paramecia_id_tracks.npy', 'wb') as outfile:
#     np.save(outfile, np.array(full_paramecia_id_tracks))
#
# with open('Prey-Tracking/full_paramecia_positions.npy', 'wb') as outfile:
#     np.save(outfile, np.array(full_paramecia_positions))
#
# with open('Prey-Tracking/full_paramecia_timestamps.npy', 'wb') as outfile:
#     np.save(outfile, np.array(full_paramecia_timestamps))


# Loading dynamics

with open('Particle-Tracking-Data/proximal_paramecia_timestamps2.npy', 'rb') as outfile:
    proximal_paramecia_timestamps = np.load(outfile)

with open('Particle-Tracking-Data/proximal_paramecia_identities2.npy', 'rb') as outfile:
    proximal_paramecia_identities = np.load(outfile)

with open('Particle-Tracking-Data/proximal_paramecia_positions.npy', 'rb') as outfile:
    proximal_paramecia_positions = np.load(outfile)

with open('Particle-Tracking-Data/fish_track_full.npy', 'rb') as outfile:
    fish_track_full = np.load(outfile)

with open('Particle-Tracking-Data/full_paramecia_positions.npy', 'rb') as outfile:
    full_paramecia_positions = np.load(outfile)

with open('Particle-Tracking-Data/full_paramecia_id_tracks.npy', 'rb') as outfile:
    full_paramecia_id_tracks = np.load(outfile)

with open('Particle-Tracking-Data/full_paramecia_timestamps.npy', 'rb') as outfile:
    full_paramecia_timestamps = np.load(outfile)


#                    Investigating Dynamics

def sort_interactions(paramecia_timestamps, paramecia_identities):
    possible_paramecia = np.unique(paramecia_identities)

    timestamp_groups = []
    timestamp_identities = []

    for p in possible_paramecia:
        timestamps = sorted(paramecia_timestamps[paramecia_identities == p])
        continuous_timestamps = []
        for i, t in enumerate(timestamps):
            if i == 0:
                continuous_timestamps.append(t)
            else:
                if t == timestamps[i-1] + 1:
                    continuous_timestamps.append(t)
                else:
                    if len(continuous_timestamps) > 0:
                        timestamp_groups.append(copy.deepcopy(continuous_timestamps))
                        timestamp_identities.append(p)
                    continuous_timestamps = []

    return timestamp_identities, timestamp_groups


timestamp_identities, timestamp_groups = sort_interactions(proximal_paramecia_timestamps, proximal_paramecia_identities)


def get_positions(timestamps, transformed_fish_position, ordered_timestamps, proximal_paramecia_positions,
                  proximal_paramecia_timestamps, proximal_paramecia_identities, identity, full_paramecia_id_tracks,
                  full_paramecia_positions, full_paramecia_timestamps):

    timestamps = np.array(timestamps).astype(int)
    start_timestamp = min(timestamps)

    # possible_paramecia_positions = proximal_paramecia_positions[proximal_paramecia_identities == identity]
    # possible_paramecia_timestamps = proximal_paramecia_timestamps[proximal_paramecia_identities == identity]
    possible_paramecia_positions = full_paramecia_positions[full_paramecia_id_tracks == identity]
    possible_paramecia_timestamps = full_paramecia_timestamps[full_paramecia_id_tracks == identity]


    paramecia_positions_before = possible_paramecia_positions[(start_timestamp - 30 <= possible_paramecia_timestamps) *
                                                              (possible_paramecia_timestamps < start_timestamp)]
    # paramecia_positions_during = possible_paramecia_positions[possible_paramecia_timestamps == timestamps]

    paramecia_positions_during = np.zeros((0, 2))
    fish_positions_during = np.zeros((0, 2))
    for t in timestamps:
        fish_position = transformed_fish_position[ordered_timestamps == t]
        fish_positions_during = np.concatenate((fish_positions_during, fish_position), axis=0)

        paramecium_position = possible_paramecia_positions[possible_paramecia_timestamps == t]
        paramecia_positions_during = np.concatenate((paramecia_positions_during, paramecium_position), axis=0)

    return paramecia_positions_before, paramecia_positions_during, fish_positions_during


def validated(paramecia_positions_before, paramecia_positions_during, fish_positions_during):
    if len(paramecia_positions_during) != len(fish_positions_during):
        return False

    distances_during = ((paramecia_positions_during[:, 0]-fish_positions_during[:, 0])**2 +
                        (paramecia_positions_during[:, 1]-fish_positions_during[:, 1])**2) ** 0.5
    if not np.any(distances_during < 100):
        return False

    if len(paramecia_positions_before) < 10:
        return False

    # Check if is static
    if np.sum(paramecia_positions_before[0]-paramecia_positions_before[-1]) < 5:
        return False



    return True

# Get deviation for each sequence

def deviation_within_sequence(prey_positions_before, prey_positions_during, fish_positions_during):
    # Distances
    distances_during = ((prey_positions_during[:, 0]-fish_positions_during[:, 0])**2 +
                        (prey_positions_during[:, 1]-fish_positions_during[:, 1])**2) ** 0.5

    deviations_during = np.zeros((0, 2))
    absolute_deviations_during = np.zeros((0))

    # Linear model of prey position
    for i, pos in enumerate(prey_positions_during):
        data = np.concatenate((prey_positions_before, prey_positions_during[:i]), axis=0)
        model = LinearRegression().fit(data[:-1], data[1:])
        # print(f"score: {model.score(data[-2:-1], pos.reshape(1, -1))}")
        prediction = model.predict(data[-2:-1])

        deviation = prey_positions_during[i:i+1] - prediction
        deviations_during = np.concatenate((deviations_during, deviation), axis=0)

        absolute_deviation = (deviation[0, 0] ** 2 + deviation[0, 1] ** 2) ** 0.5

        # Make positive or negative, depending on whether closer or further away from fish.
        if (((prediction[0, 0] - fish_positions_during[i, 0]) ** 2 +
            (prediction[0, 1] - fish_positions_during[i, 1]) ** 2) ** 0.5) > (((pos[0] - fish_positions_during[i, 0]) ** 2 +
            (pos[1] - fish_positions_during[i, 1]) ** 2) ** 0.5):
            absolute_deviation *= -1

        absolute_deviation = np.array([absolute_deviation])

        absolute_deviations_during = np.concatenate((absolute_deviations_during, absolute_deviation), axis=0)

    return deviations_during, distances_during, absolute_deviations_during


all_deviations = np.zeros((0, 2))
all_distances = np.zeros((0))
all_timestamps = []
absolute_deviations_during = np.zeros((0))

for par, timestamps in zip(timestamp_identities, timestamp_groups):

    paramecia_positions_before, paramecia_positions_during, fish_positions_during = \
        get_positions(timestamps, transformed_fish_position, ordered_timestamps, proximal_paramecia_positions,
                  proximal_paramecia_timestamps, proximal_paramecia_identities, par, full_paramecia_id_tracks,
                      full_paramecia_positions, full_paramecia_timestamps)

    if validated(paramecia_positions_before, paramecia_positions_during, fish_positions_during):
        deviations, distances, absolute_deviation = deviation_within_sequence(paramecia_positions_before, paramecia_positions_during, fish_positions_during)
        all_deviations = np.concatenate((all_deviations, deviations), axis=0)
        all_distances = np.concatenate((all_distances, distances), axis=0)
        absolute_deviations_during = np.concatenate((absolute_deviations_during, absolute_deviation), axis=0)
        all_timestamps = all_timestamps + timestamps

    else:
        pass

all_timestamps = np.array(all_timestamps)

# Get fish distances moved

fish_positions_in_data = np.zeros((0, 2))

for t in all_timestamps:
    positions = fish_track_full[fish_timestamps_full == t]
    fish_positions_in_data = np.concatenate((fish_positions_in_data, positions), axis=0)

difference = fish_positions_in_data[1:]-fish_positions_in_data[:-1]

fish_distances_moved = ((difference[:, 0] ** 2) + (difference[:, 1] ** 2)) ** 0.5
fish_distances_moved = np.concatenate((np.array([0]), fish_distances_moved), axis=0)


#              Unit Conversion

# Original - 80fps
# Reduced - Every 8 frames (as these are darkfield), so 10fps. Each frame is 0.1s
# Displaypxpermm 12.987

absolute_deviations = ((all_deviations[:, 0]**2) + (all_deviations[:, 1]**2)) ** 0.5
displacement_mm = absolute_deviations/12.987
fish_distances_moved = fish_distances_moved/12.987
all_distances = all_distances/12.987
absolute_deviations_during = absolute_deviations_during/12.987

#              Data Visualisation

plt.scatter(all_distances, displacement_mm, alpha=0.5)
plt.xlabel("Distance from fish (mm)")
plt.ylabel("Deviation from trajectory (mm)")
plt.show()

plt.scatter(fish_distances_moved, displacement_mm, alpha=0.5)
plt.xlabel("Fish distance moved (mm)")
plt.ylabel("Deviation from trajectory (mm)")
plt.show()

plt.scatter(all_distances, absolute_deviations_during, alpha=0.5)
plt.xlabel("Distance from fish (mm)")
plt.ylabel("Deviation from trajectory (mm)")
plt.show()

plt.scatter(fish_distances_moved, absolute_deviations_during, alpha=0.5)
plt.xlabel("Fish distance moved (mm)")
plt.ylabel("Deviation from trajectory (mm)")
plt.show()

# plt.scatter(all_distances, displacement_mm, alpha=0.5)
# plt.xlabel("Distance from fish (mm)")
# plt.ylabel("Deviation from trajectory (mm)")
# plt.show()
#
# plt.scatter(fish_distances_moved, displacement_mm, alpha=0.5)
# plt.xlabel("Fish distance moved (mm)")
# plt.ylabel("Deviation from trajectory (mm)")
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(all_distances, displacement_mm, fish_distances_moved)
plt.show()


#           Cut Down


all_distances = all_distances[displacement_mm < 2.5]
fish_distances_moved = fish_distances_moved[displacement_mm < 2.5]
absolute_deviations_during = absolute_deviations_during[displacement_mm < 2.5]
displacement_mm = displacement_mm[displacement_mm < 2.5]


fish_distances_moved = fish_distances_moved[all_distances < 4]
displacement_mm = displacement_mm[all_distances < 4]
absolute_deviations_during = absolute_deviations_during[all_distances < 4]
all_distances = all_distances[all_distances < 4]


plt.scatter(all_distances, displacement_mm, alpha=0.5)
plt.xlabel("Distance from fish (mm)")
plt.ylabel("Deviation from trajectory (mm)")
plt.show()

plt.scatter(fish_distances_moved, displacement_mm, alpha=0.5)
plt.xlabel("Fish distance moved (mm)")
plt.ylabel("Deviation from trajectory (mm)")
plt.show()

plt.scatter(all_distances, absolute_deviations_during, alpha=0.5)
plt.xlabel("Distance from fish (mm)")
plt.ylabel("Deviation from trajectory (mm)")
plt.show()

plt.scatter(fish_distances_moved, absolute_deviations_during, alpha=0.5)
plt.xlabel("Fish distance moved (mm)")
plt.ylabel("Deviation from trajectory (mm)")
plt.show()
