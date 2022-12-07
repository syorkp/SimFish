import numpy as np
import matplotlib.pyplot as plt


energy_states = np.linspace(0, 1, 1000)

r_vals = 1 - energy_states
g_vals = energy_states
b_vals = 0 * energy_states

rgb_vals = np.concatenate((np.expand_dims(r_vals, 1), np.expand_dims(g_vals, 1), np.expand_dims(b_vals, 1)), axis=1)
rgb_vals = np.expand_dims(rgb_vals, 0)
rgb_vals = np.tile(rgb_vals, (100, 1, 1))
plt.imshow(rgb_vals)
plt.show()

