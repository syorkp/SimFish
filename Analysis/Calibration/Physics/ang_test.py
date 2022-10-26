import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([1, 1.5])
v2 = np.array([-1.5, 2])
v3 = np.array([-1, -1])
v4 = np.array([2, -1.5])

all_points = np.concatenate((np.expand_dims(v1, 0), np.expand_dims(v2, 0), np.expand_dims(v3, 0), np.expand_dims(v4, 0)), axis=0)
all_angles = np.linspace(-np.pi + 1, np.pi, 10)
distances = (all_points[:, 0] ** 2 + all_points[:, 1] ** 2) ** 0.5
all_points[:, 0] /= distances
all_points[:, 1] /= distances

angle = np.arctan(all_points[:, 1]/all_points[:, 0])

for i, a in enumerate(angle):
    if all_points[i, 0] < 0 and all_points[i, 1] < 0:
        # Generates postiive angle from left x axis clockwise.
        # print("UL quadrent")
        angle[i] += np.pi
    elif all_points[i, 1] < 0:
        # Generates negative angle from right x axis anticlockwise.
        # print("UR quadrent.")
        angle[i] += (np.pi * 2)
    elif all_points[i, 0] < 0:
        # Generates negative angle from left x axis anticlockwise.
        # print("BL quadrent.")
        angle[i] += np.pi

fish_angs = np.array([0.9, 3, 10, -5])
fish_angs = (fish_angs % (2 * np.pi))

fish_points_y = np.sin(all_angles)
fish_points_x = np.cos(all_angles)
fish_points = np.concatenate((np.expand_dims(fish_points_x, 1), np.expand_dims(fish_points_y, 1)), axis=1)

too_high = angle > np.pi
angle[too_high] -= np.pi * 2
too_low = angle < -np.pi
angle[too_low] += np.pi * 2

too_high = fish_angs > np.pi
fish_angs[too_high] -= np.pi * 2
too_low = fish_angs < -np.pi
fish_angs[too_low] += np.pi * 2

# plt.scatter(all_points[:, 0], all_points[:, 1])
plt.scatter(fish_points[:, 0], fish_points[:, 1])
# for i, a in enumerate(angle):
#     # plt.text(all_points[i, 0], all_points[i, 1], str(a))
#     plt.text(fish_points[i, 0], fish_points[i, 1], str(fish_angs[i]))
#
plt.show()

all_vectors = fish_points[1:] - fish_points[:-1]
all_distances = (all_vectors[:1, 0] ** 2 + all_vectors[:, 1] ** 2) ** 0.5

all_angles = np.concatenate((all_angles, all_angles))
all_differences = np.absolute(all_angles[1:] - all_angles[:-1])
out = all_differences > np.pi
all_differences[out] -= 2 * np.pi
all_differences = np.absolute(all_differences)

# p = np.polyfit(all_distances, all_differences, 1)
#
# plt.scatter(all_distances, all_differences)
# plt.show()