import numpy as np
import matplotlib.pyplot as plt

# with open('SNR/distances.npy', 'rb') as outfile:
#     distances = np.load(outfile)
#
#
# with open('SNR/snrs.npy', 'rb') as outfile:
#     snrs = np.load(outfile)
#
# distances = np.expand_dims(distances, 1)
# distances = np.concatenate((distances, distances), axis=1)
# distances = distances.flatten()
#
# distances = distances[~np.isnan(snrs)]
# snrs = snrs[~np.isnan(snrs)]
#
# z = np.polyfit(distances, snrs, 1)
# p = np.poly1d(z)
#
# plt.scatter(distances, snrs)
# plt.plot(distances, p(distances), c="r")
#
# plt.show()


with open('SNR/distances2.npy', 'rb') as outfile:
    distances = np.load(outfile)

with open('SNR/red_fail_left.npy', 'rb') as outfile:
    red_fail_left = np.load(outfile)

with open('SNR/red_fail_right.npy', 'rb') as outfile:
    red_fail_right = np.load(outfile)

with open('SNR/uv_fail_left.npy', 'rb') as outfile:
    uv_fail_left = np.load(outfile)

with open('SNR/uv_fail_right.npy', 'rb') as outfile:
    uv_fail_right = np.load(outfile)

with open('SNR/red2_fail_left.npy', 'rb') as outfile:
    red2_fail_left = np.load(outfile)

with open('SNR/red2_fail_right.npy', 'rb') as outfile:
    red2_fail_right = np.load(outfile)

# distances = np.expand_dims(distances, 1)
# distances = np.concatenate((distances, distances), axis=1)
# distances = distances.flatten()

# Red

distances1 = distances[~np.isnan(red_fail_left)]
red_fail_left = red_fail_left[~np.isnan(red_fail_left)]
distances2 = distances[~np.isnan(red_fail_right)]
red_fail_right = red_fail_right[~np.isnan(red_fail_right)]

distances1 = np.concatenate((distances1, distances2), axis=0)
red_fail_left = np.concatenate((red_fail_left, red2_fail_right), axis=0)

z = np.polyfit(distances1, red_fail_left, 1)
p = np.poly1d(z)
plt.plot(distances1, p(distances1), c="r")

plt.scatter(distances1, red_fail_left, alpha=0.2)
plt.show()


# UV

distances1 = distances[~np.isnan(uv_fail_left)]
uv_fail_left = uv_fail_left[~np.isnan(uv_fail_left)]
distances2 = distances[~np.isnan(uv_fail_right)]
uv_fail_right = uv_fail_right[~np.isnan(uv_fail_right)]

distances1 = np.concatenate((distances1, distances2), axis=0)
uv_fail_left = np.concatenate((uv_fail_left, uv_fail_right), axis=0)

z = np.polyfit(distances1, uv_fail_left, 1)
p = np.poly1d(z)
plt.plot(distances1, p(distances1), c="r")

plt.scatter(distances1, uv_fail_left, alpha=0.2)
plt.show()

# Red 2

distances1 = distances[~np.isnan(red2_fail_left)]
red2_fail_left = red2_fail_left[~np.isnan(red2_fail_left)]
distances2 = distances[~np.isnan(red2_fail_right)]
red2_fail_right = red2_fail_right[~np.isnan(red2_fail_right)]

distances1 = np.concatenate((distances1, distances2), axis=0)
red2_fail_left = np.concatenate((red2_fail_left, red2_fail_right), axis=0)

z = np.polyfit(distances1, red2_fail_left, 1)
p = np.poly1d(z)
plt.plot(distances1, p(distances1), c="r")

plt.scatter(distances1, red2_fail_left, alpha=0.2)
plt.show()

x = True
# z = np.polyfit(distances, snrs, 1)
# p = np.poly1d(z)
# plt.plot(distances, p(distances), c="r")
#
# plt.show()



