import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def get_intensity_decay(x_min, x_max, y_min, y_max, image):
    line_points = []
    for y in range(y_min, y_max):
        plane_points = []
        for x in range(x_min, x_max):
            plane_points.append(image[x, y])
        plane_points = np.array(plane_points)
        line_points.append(np.sum(plane_points, axis=0))
    return np.array(line_points)


def plot_intensity_data(lines):
    plt.plot(np.linspace(0, 30, len(lines)), lines[:, 0], color="r")
    plt.plot(np.linspace(0, 30, len(lines)), lines[:, 1], color="g")
    plt.plot(np.linspace(0, 30, len(lines)), lines[:, 2], color="b")

    distance = np.array(range(0, 3000))  # Measure of distance from fish at every point.
    scatter = np.exp(-0.0006 * distance) * 70000
    plt.plot(np.linspace(0, 30, len(scatter)), scatter, color="y")

    plt.show()


def plot_decay(decay_constant=0.001):
    distance = np.array(range(0, 3000))  # Measure of distance from fish at every point.
    scatter = np.exp(-decay_constant * distance) * 70000
    plt.plot(distance, scatter)
    plt.show()


im = Image.open('Image/AVG_Measurements.png')
pix = im.load()

line = get_intensity_decay(348, 632, 26, 955, pix)
plot_intensity_data(line)
# plot_decay()



x = True
