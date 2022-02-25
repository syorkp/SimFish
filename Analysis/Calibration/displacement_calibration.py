import matplotlib.pyplot as plt
import numpy as np

dists = np.linspace(0, 2, 100)
scaled = np.exp(-dists)

plt.plot(dists, scaled)
plt.show()




