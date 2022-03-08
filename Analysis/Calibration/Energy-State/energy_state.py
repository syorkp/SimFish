import numpy as np
import matplotlib.pyplot as plt


def display_trajectory(B):
    energy_levels = np.linspace(0, 1, 1000)
    mul_factors = [trajectory(e, B) for e in energy_levels]
    plt.plot(energy_levels, mul_factors)
    plt.show()


def trajectory(energy_level, B=9):
    A = 1/np.exp(B)
    return A * np.exp(B*energy_level)


def display_intake(C):
    energy_levels = np.linspace(0, 1, 1000)
    mul_factors = [intake(e, C) for e in energy_levels]
    plt.plot(energy_levels, mul_factors)
    plt.show()


def intake(energy_level, C=2.5):
    # return (1-energy_level) * (-np.exp(C*energy_level))
    return np.exp(-C*energy_level)


def model_episode(consumptions, impulses, angles, ca, ci, cc, baseline, duration):
    energy_level = [1]
    for i in range(duration):
        Ei2 = energy_level[i] + (intake(energy_level[i])*consumptions[i]*cc) - (trajectory(energy_level[i])*((ca*angles[i]+ci*impulses[i]) + baseline))
        energy_level.append(Ei2)
    plt.plot([i for i in range(duration)], energy_level[1:])
    plt.show()


ca = 0.01
ci = 0.01
cc = 0.2
baseline_decrease = 0.001
B = np.linspace(0, 10, 10)  # GO with 5 for now.
C = np.linspace(0, 10, 10)  # GO with 5 for now.

captures = np.random.choice([0, 1], size=1000, p=[1-0.01, 0.01]).astype(float)
# captures = np.zeros((100000))


i = np.random.random(1000) * 10.0
a = np.random.random(1000) * (np.pi/5)

model_episode(captures, i, a, ca, ci, cc, baseline_decrease, 1000)



