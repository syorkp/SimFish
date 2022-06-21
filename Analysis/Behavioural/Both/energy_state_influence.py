import matplotlib.pyplot as plt
import numpy as np

from Analysis.Behavioural.Tools.get_repeated_data_parameter import get_parameter_across_trials


def show_action_histograms(action_data, energy_state_data):
    flattened_action_data = np.concatenate(action_data, axis=0)
    flattened_energy_state_data = np.concatenate(energy_state_data, axis=0)

    for a in range(0, 10):
        energy = flattened_energy_state_data[flattened_action_data == a]
        plt.hist(energy, bins=20)
        plt.title(f"{a}")
        plt.show()


action_data = get_parameter_across_trials(f"dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 20, "action")
energy_state_data = get_parameter_across_trials(f"dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 20, "energy_state")




# TODO: Make raster plot display.
# TODO: Create probability of actions gridplot at different energy states.