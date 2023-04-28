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


if __name__ == "__main__":
    action_data = get_parameter_across_trials(f"dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic", 100, "action")
    energy_state_data = get_parameter_across_trials(f"dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic", 100, "energy_state")

    show_action_histograms(action_data, energy_state_data)
