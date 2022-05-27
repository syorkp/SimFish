import matplotlib.pyplot as plt
import numpy as np

from Analysis.Behavioural.Tools.get_repeated_data_parameter import get_parameter_across_trials


action_data = get_parameter_across_trials(f"dqn_scaffold_14-2", "Behavioural-Data-Free", "Naturalistic", 10, "action")
energy_state_data = get_parameter_across_trials(f"dqn_scaffold_14-2", "Behavioural-Data-Free", "Naturalistic", 10, "energy_state")

flattened_action_data = np.concatenate(action_data, axis=0)
flattened_energy_state_data = np.concatenate(energy_state_data, axis=0)

for a in range(0, 10):
    energy = flattened_energy_state_data[flattened_action_data == a]
    plt.hist(energy)
    plt.title(f"{a}")
    plt.show()


# TODO: Make raster plot display.
# TODO: Create probability of actions gridplot at different energy states.