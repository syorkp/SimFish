from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


im = Image.open('img.png')
pix = im.load()

dims = (593, 310)

swapped_pixels = np.zeros((dims[0], dims[1], 4))

for i in range(dims[0]):
    for j in range(dims[1]):
        pixel = pix[i, j]
        swapped_pixels[i, j] = pixel

inverted_pixels = np.absolute(255.0 - swapped_pixels)

relevant = inverted_pixels[:, :, :3]

c1 = relevant[:, :, 0]
c2 = relevant[:, :, 1]
c3 = relevant[:, :, 2]

# Convert remaining dots to pixels
def pixels_to_points(rgb):
    dims = rgb.shape
    data_points = []

    for c in range(dims[2]):
        for i in range(dims[0]):
            for j in range(dims[1]):
                pixel = int(rgb[i, j, c])
                for p in range(pixel):
                    data_points.append([i, j])

    return np.array(data_points)

points = pixels_to_points(relevant)

x_points = points[:, 0]
y_points = np.abs(600 - points[:, 1])

heatmap, xedges, yedges = np.histogram2d(x_points, y_points, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

model = np.polyfit(x_points, y_points, 1)
p = np.poly1d(model)

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
widths = range(int(xedges[0]), int(xedges[-1]))
plt.plot(widths, p(widths), color="r")

plt.show()



