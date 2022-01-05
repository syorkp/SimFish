import matplotlib.pyplot as plt
import numpy as np
import math


def db_compute_n(angular_size, max_separation=1, width=1500, height=1500):
    max_dist = (width ** 2 + height ** 2) ** 0.5
    theta_separation = math.asin(max_separation / max_dist)
    n = (angular_size / theta_separation) / 2
    return int(n)


sizes = np.linspace(0, 50, 1000)

plt.plot(sizes, [db_compute_n(s) for s in sizes])
plt.show()



