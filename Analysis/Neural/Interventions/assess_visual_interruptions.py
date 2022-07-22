import numpy as np
import matplotlib.pyplot as plt
from Analysis.Neural.Visualisation.display_many_neurons import plot_traces
from Analysis.load_data import load_data


data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Long-Interruptions", "Naturalistic-1")
data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")

rnn_data = data["rnn_state_actor"][:300, 0, 0, :]
rnn_data = np.swapaxes(rnn_data, 0, 1)

# plt.plot(data["salt"])
# plt.show()
plot_traces(rnn_data)

