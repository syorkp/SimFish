import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data

data = load_data("continuous_extra_rnn_learning-1", "Continuous_Action_Mapping", "Environment-1")

all_impulses = data["impulse"]
all_angles = np.absolute(data["angle"])

plt.scatter(all_impulses, all_angles, alpha=0.1)
plt.show()
