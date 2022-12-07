import numpy as np
import matplotlib.pyplot as plt

xp, yp = np.arange(1500), np.arange(1500)

i, j = xp[:, None], yp[None, :]
decay_rate = 0.001

x = 750
y = 750
positional_mask = (((x - i) ** 2 + (y - j) ** 2) ** 0.5)  # Measure of distance from fish at every point.

desired_scatter = np.exp(-decay_rate * positional_mask)
plt.imshow(desired_scatter)
plt.show()

distance = np.linspace(0, 75, 750)
plt.plot(distance, desired_scatter[750, 750:])
plt.xlabel("Distance (mm)")
plt.ylabel("Proportion of light let through")
plt.show()

