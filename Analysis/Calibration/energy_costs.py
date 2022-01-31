import numpy as np
import matplotlib.pyplot as plt


def intake_scale(energy_level, trajectory_B, trajectory_B2):
    """Provides nonlinear scaling for consumption reward and energy level change for new simulation"""
    return trajectory_B2 * np.exp(-trajectory_B * energy_level)


def action_scale(energy_level, trajectory_A, trajectory_A2):
    """Provides nonlinear scaling for action penalty and energy level change for new simulation"""
    return trajectory_A2 * np.exp(trajectory_A * energy_level)


def calculate_reward_and_cost(energy_level, consumption=True, impulse=5.0, angle=1.0):
    # Constants
    baseline_decrease = 0.005
    ca = 0.01
    ci = 0.01
    trajectory_A = 5.0
    trajectory_B = 2.5
    trajectory_A2 = 1/np.exp(trajectory_A)
    trajectory_B2 = 1/np.exp(trajectory_B)
    action_reward_scaling = 10
    consumption_reward_scaling = 1000000

    unscaled_consumption = 1.0 * consumption
    unscaled_energy_use = ci * impulse + ca * angle + baseline_decrease
    energy_level += unscaled_consumption - unscaled_energy_use

    # Nonlinear reward scaling
    intake_s = intake_scale(energy_level, trajectory_B, trajectory_B2)
    action_s = action_scale(energy_level, trajectory_A, trajectory_A2)
    energy_intake = (intake_s * unscaled_consumption)
    energy_use = (action_s * unscaled_energy_use)
    reward_1, reward_2 = (energy_intake * consumption_reward_scaling), - (energy_use * action_reward_scaling)

    return reward_1, reward_2


results = np.array([[calculate_reward_and_cost(i)] for i in np.linspace(0, 1.0, 1000)])


plt.plot(np.linspace(0, 1.0, 1000), results[:, 0, 0])
plt.show()

plt.plot(np.linspace(0, 1.0, 1000), results[:, 0, 1])
plt.show()

