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

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


file = "SNR2"

with open(f'{file}/distances2.npy', 'rb') as outfile:
    distances = np.load(outfile)

with open(f'{file}/red_fail_left.npy', 'rb') as outfile:
    red_fail_left = np.load(outfile)

with open(f'{file}/red_fail_right.npy', 'rb') as outfile:
    red_fail_right = np.load(outfile)

with open(f'{file}/uv_fail_left.npy', 'rb') as outfile:
    uv_fail_left = np.load(outfile)

with open(f'{file}/uv_fail_right.npy', 'rb') as outfile:
    uv_fail_right = np.load(outfile)

with open(f'{file}/red2_fail_left.npy', 'rb') as outfile:
    red2_fail_left = np.load(outfile)

with open(f'{file}/red2_fail_right.npy', 'rb') as outfile:
    red2_fail_right = np.load(outfile)

# distances = np.expand_dims(distances, 1)
# distances = np.concatenate((distances, distances), axis=1)
# distances = distances.flatten()

# Red
if file != "SNR":
    red_fail_left = np.array([i.get() for i in red_fail_left])
    red_fail_right = np.array([i.get() for i in red_fail_right])
    uv_fail_left = np.array([i.get() for i in uv_fail_left])
    uv_fail_right = np.array([i.get() for i in uv_fail_right])
    red2_fail_left = np.array([i.get() for i in red2_fail_left])
    red2_fail_right = np.array([i.get() for i in red2_fail_right])


def remove_distances_over(distances, red_fail_left, red_fail_right, uv_fail_left, uv_fail_right, red2_fail_left,
                          red2_fail_right, distance=1500):
    red_fail_left = red_fail_left[distances < distance]
    red_fail_right = red_fail_right[distances < distance]
    uv_fail_left = uv_fail_left[distances < distance]
    uv_fail_right = uv_fail_right[distances < distance]
    red2_fail_left = red2_fail_left[distances < distance]
    red2_fail_right = red2_fail_right[distances < distance]
    distances = distances[distances < distance]
    return distances, red_fail_left, red_fail_right, uv_fail_left, uv_fail_right, red2_fail_left, red2_fail_right


distances, red_fail_left, red_fail_right, uv_fail_left, uv_fail_right, red2_fail_left, red2_fail_right = remove_distances_over(distances, red_fail_left, red_fail_right, uv_fail_left, uv_fail_right, red2_fail_left,
                          red2_fail_right)
distances = distances/10

distances1 = distances[~np.isnan(red_fail_left)]
red_fail_left = red_fail_left[~np.isnan(red_fail_left)]
distances2 = distances[~np.isnan(red_fail_right)]
red_fail_right = red_fail_right[~np.isnan(red_fail_right)]

distances1 = np.concatenate((distances1, distances2), axis=0)
red_fail_left = np.concatenate((red_fail_left, red2_fail_right), axis=0)

z = np.polyfit(distances1, red_fail_left, 1)
p = np.poly1d(z)
print(p)

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

uv_fail_left = uv_fail_left[distances1 < 40]
distances1 = distances1[distances1 < 40]


# distances1 = distances1[uv_fail_left != 0]
# uv_fail_left = uv_fail_left[uv_fail_left != 0]

# hist1, _ = np.histogram(distances1,range=(0,1500), bins=100)
# hist2, _ = np.histogram(distances3,range=(0, 1500), bins=100)
#
# diff = np.absolute(hist1 - hist2)
# plt.plot(diff)
# plt.show()

# plt.hist(distances3, bins=100)
# plt.show()

z = np.polyfit(distances1, uv_fail_left, 1)
p = np.poly1d(z)

print(p)

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
print(p)

plt.plot(distances1, p(distances1), c="r")
plt.scatter(distances1, red2_fail_left, alpha=0.2)
plt.show()

x = True
# z = np.polyfit(distances, snrs, 1)
# p = np.poly1d(z)
# plt.plot(distances, p(distances), c="r")
#
# plt.show()



