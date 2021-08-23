import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data

data = load_data("scaffold_test-19", "Checking_Observation", "Environment-1")

all_impulses = data["impulse"]
all_angles = np.absolute(data["angle"])

consumption_timestamps = [i for i, c in enumerate(data["consumed"]) if c == 1]

plt.scatter(all_impulses, all_angles, alpha=0.1)
plt.show()

plt.plot(all_impulses[430:473])
plt.show()

plt.plot(all_angles[430:473])
plt.show()

x = True
