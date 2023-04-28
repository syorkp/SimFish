import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path of file 1: smb://ad.ucl.ac.uk/groupfolders/DBIO_Bianco_Lab4/Joanna/Data2/Ablations/2017/2017-04/2017-04-06%20ablations/freeswimming/ab1
# Path of file 2: 'Y:\Joanna\Data\Ablations\2017\2017-05\2017-05-23\f1\f1_pre_freeswim\';
# Path of file 3: smb://ad.ucl.ac.uk/groupfolders/DBIO_Bianco_Lab4/Joanna/Data2/Ablations/2017/2017-06/2017-06-13/f2/f2_freeswim

def load_data(file_name):
    with h5py.File(f"./Motor-Noise-Data/{file_name}") as f:
        dstruct = f.get("aq")

        ori = dstruct["ori"]
        inmiddle = dstruct["inmiddle"]
        visstim = dstruct["visstim"]
        data = dstruct["data"]
        timebase = dstruct["timebase"]
        bouts = dstruct["bouts"]

        ori = np.array(ori)
        inmiddle = np.array(inmiddle)
        visstim = np.array(visstim)
        data = np.array(data)
        timebase = np.array(timebase)
        bouts = np.array(bouts)

    # Remove entries not in middle
    ori = ori[0, inmiddle[0] > 0]
    data = data[:, inmiddle[0] > 0]
    timebase = timebase[0, inmiddle[0] > 0]

    # Extract only fields we want
    position_frames = data[0, :]
    x_position = data[1, :]
    y_position = data[2, :]

    # ori = ori[0]
    # timebase = timebase[0]

    return visstim, position_frames, x_position, y_position, ori, timebase, bouts


def flatten_orientation(ori):
    ori = ori + 180
    d_ori = ori[1:] - ori[:-1]
    # plt.plot([i for i in range(len(ori))], ori)
    # plt.show()

    transformed_ori = []
    current_adjustment = 0

    for i, o in enumerate(ori):
        if i == 0:
            transformed_ori.append(o)
        else:
            if d_ori[i-1] > 300:
                current_adjustment -= 360
            elif d_ori[i-1] < -300:
                current_adjustment += 360
            transformed_ori.append(o + current_adjustment)
    # plt.plot([i for i in range(len(transformed_ori))], transformed_ori)
    # plt.show()

    transformed_ori = np.array(transformed_ori)
    transformed_ori = transformed_ori + 2880

    return transformed_ori


def get_distance_and_ori_change_within_frames(start_frame, end_frame, position_frames, x_position, y_position, ori, timebase):
    indices = (start_frame <= position_frames) * (position_frames <= end_frame)
    x_position_during = x_position[indices]
    y_position_during = y_position[indices]
    position_during = np.concatenate((np.expand_dims(x_position_during, 1), np.expand_dims(y_position_during, 1)), axis=1)
    orientation_during = ori[indices]
    # timebase = timebase[indices]

    distance = ((x_position_during[1:] - x_position_during[:-1]) ** 2 + (y_position_during[1:] - y_position_during[:-1])) ** 0.5
    orientation = orientation_during[1:] - orientation_during[:-1]

    return distance, orientation, position_during


def get_bout_parameters(frame_start, frame_end, bout_data, position_frames, x_position, y_position):
    indexed_bouts = (frame_start <= bout_data[0, :]) * (frame_end >= bout_data[0, :])
    bouts = bout_data[:, indexed_bouts]
    times = bouts[2]
    delta_angle = bouts[5]


    bout_distances = []
    # Compute distances moved
    for b in range(bouts.shape[1]):
        t_start, t_end = bouts[0, b], bouts[1, b]

        indices = (t_start <= position_frames) * (position_frames <= t_end)

        x_positions = x_position[indices]
        y_positions = y_position[indices]

        if len(x_positions) > 0:

            x_position_start = x_positions[0]
            y_position_start = y_positions[0]
            x_position_end = x_positions[-1]
            y_position_end = y_positions[-1]

            distance = ((x_position_end - x_position_start) ** 2 + (y_position_end - y_position_start)) ** 0.5

        else:
            distance = 0

        bout_distances.append(distance)

    return times, delta_angle, bout_distances


# Physical parameters: pxpermm - 24.8, 12.987; rate - 700fps/m; radius - 280
# For translation to mm, divide by 12.987 as in videos

visstim, position_frames, x_position, y_position, ori, timebase, bouts = load_data("aq.mat")

# ori = flatten_orientation(ori)

min_frame = min(position_frames)
max_frame = max(position_frames)

directions = []

durations = []
angles = []
distances = []


for trial in range(visstim.shape[1]):
    start_frame = visstim[3, trial]
    end_frame = visstim[4, trial]
    if visstim[2, trial] == 0:
        direction = "L"
    else:
        direction = "R"

    if start_frame > min_frame and end_frame < max_frame and visstim[1, trial] == 3 and visstim[2, trial] == 1:
        distance, orientation, position_during = get_distance_and_ori_change_within_frames(start_frame, end_frame, position_frames, x_position, y_position, ori, timebase)
        bout_durations, bout_delta_angle, bout_distances = get_bout_parameters(start_frame, end_frame, bouts, position_frames, x_position, y_position)

        if len(bout_durations) > 1:
            durations.append(bout_durations[1])
            angles.append(bout_delta_angle[1])
            distances.append(bout_distances[1])

        if position_during.shape[0] > 0:
            print(f"Trial: {trial}, length: {len(distance)}")
            motion_vector = position_during[0, :] - position_during[-1, :]
            directions.append(motion_vector)

            # plt.plot(range(len(distance)), distance)
            # plt.show()
            #
            # plt.plot(range(len(orientation)), orientation)
            # plt.show()

            plt.scatter(position_during[:, 0], position_during[:, 1], c=range(len(position_during[:, 0])), alpha=0.02)
            plt.title(f"{trial}, {direction}")
            plt.show()

plt.hist(durations, bins=30)
plt.show()

plt.hist(angles, bins=30)
plt.show()

plt.hist(distances, bins=30)
plt.show()
