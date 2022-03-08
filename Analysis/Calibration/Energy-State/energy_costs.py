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
    baseline_decrease = 0.0005
    ca = 0.0001
    ci = 0.0001
    trajectory_A = 5.0
    trajectory_B = 2.5
    trajectory_A2 = 1/np.exp(trajectory_A)
    trajectory_B2 = 1/np.exp(trajectory_B)
    action_reward_scaling = 10
    consumption_reward_scaling = 10000000

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


def get_trajectory(consumptions, impulses, angles, ci, ca, consumption_energy_gain, baseline_decrease, trajectory_A,
                   trajectory_B, action_reward_scaling, consumption_reward_scaling,  num_steps=1000):

    energy_level = 1.0

    trajectory_A2 = 1/np.exp(trajectory_A)
    trajectory_B2 = 1/np.exp(trajectory_B)

    reward = []
    action_penalty = []
    energy_log = []

    for step in range(num_steps):
        # Energy
        unscaled_consumption = consumption_energy_gain * consumptions[step]
        unscaled_energy_use = ci * impulses[step] + ca * angles[step] + baseline_decrease
        energy_level += unscaled_consumption - unscaled_energy_use
        if energy_level > 1.0:
            energy_level = 1.0
        energy_log.append(energy_level)

        # Reward
        intake_s = intake_scale(energy_level, trajectory_B, trajectory_B2)
        action_s = action_scale(energy_level, trajectory_A, trajectory_A2)
        energy_intake = (intake_s * unscaled_consumption)
        energy_use = (action_s * unscaled_energy_use)

        reward.append((energy_intake * consumption_reward_scaling) - (energy_use * action_reward_scaling))
        action_penalty.append(energy_use)


    print(f"Total reward: {np.sum(reward)}")

    steps = [i for i in range(num_steps)]

    plt.plot(steps, reward)
    plt.show()

    plt.plot(steps, action_penalty)
    plt.show()

    plt.plot(steps, energy_log)
    plt.show()


def get_returns_from_investment(ci, ca, baseline_decrease, trajectory_A, trajectory_B, consumption_energy_gain,
                                action_reward_scaling, consumption_reward_scaling, impulses, angles):
    """Compute action reward penalties, and consumption returns to find level where worth it."""
    energy_level = 1.0
    num_steps = len(impulses)

    trajectory_A2 = 1/np.exp(trajectory_A)
    trajectory_B2 = 1/np.exp(trajectory_B)

    total_reward = 0

    for step in range(num_steps):
        # Energy
        if step == num_steps - 1:
            unscaled_consumption = consumption_energy_gain * 1
        else:
            unscaled_consumption = 0
        unscaled_energy_use = ci * impulses[step] + ca * angles[step] + baseline_decrease
        energy_level += unscaled_consumption - unscaled_energy_use
        if energy_level > 1.0:
            energy_level = 1.0

        # Reward
        intake_s = intake_scale(energy_level, trajectory_B, trajectory_B2)
        action_s = action_scale(energy_level, trajectory_A, trajectory_A2)
        energy_intake = (intake_s * unscaled_consumption)
        energy_use = (action_s * unscaled_energy_use)

        total_reward += ((energy_intake * consumption_reward_scaling) - (energy_use * action_reward_scaling))

    print(total_reward)


# Chosen parameters

ci = 0.0001
ca = 0.0001
baseline_decrease = 0.0003
trajectory_A = 5.0
trajectory_B = 2.5
consumption_energy_gain = 1.0
action_reward_scaling = 1942
consumption_reward_scaling = 50000


# Computing return from investment
steps = 500
impulses = np.random.uniform(4, 6, steps)
angles = np.random.uniform(-1, 1, steps)
get_returns_from_investment(ci, ca, baseline_decrease, trajectory_A, trajectory_B, consumption_energy_gain,
                                action_reward_scaling, consumption_reward_scaling, impulses, angles)


# Modelling an episode

steps = 1000
consumptions = np.random.choice([0, 1], steps, p=[1-0.002, 0.002])
# consumptions = np.random.choice([0, 1], steps, p=[1-0.0, 0.0])
impulses = np.random.uniform(0, 20, steps)
angles = np.random.uniform(-1, 1, steps)

get_trajectory(consumptions, impulses, angles, ci, ca, consumption_energy_gain, baseline_decrease, trajectory_A,
                   trajectory_B, action_reward_scaling, consumption_reward_scaling,  num_steps=steps)

# results = np.array([[calculate_reward_and_cost(i)] for i in np.linspace(0, 1.0, 1000)])
#
#
# plt.plot(np.linspace(0, 1.0, 1000), results[:, 0, 0])
# plt.show()
#
# plt.plot(np.linspace(0, 1.0, 1000), results[:, 0, 1])
# plt.show()

