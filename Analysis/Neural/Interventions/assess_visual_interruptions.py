import numpy as np
import matplotlib.pyplot as plt

from Analysis.Neural.Visualisation.display_many_neurons import plot_traces
from Analysis.load_data import load_data
from Analysis.Neural.Systems.plot_pca_trajectory import plot_pca_trajectory


data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Long-Interruptions", "Naturalistic-3")
# data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")

rnn_data = data["rnn_state_actor"][:, 0, 0, :]
rnn_data = np.swapaxes(rnn_data, 0, 1)

# plt.plot(data["salt"])
# plt.show()
# plot_traces(rnn_data)
interruptions = [i for i in range(len(data["consumed"])) if i > 100 and i % 10 == 0]

consumption_points = [i for i in range(len(data["consumed"])) if data["consumed"][i]]
plot_pca_trajectory(rnn_data, timepoints_to_label=[99]+consumption_points)


