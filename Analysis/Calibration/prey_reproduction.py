import numpy as np
import matplotlib.pyplot as plt


def reproduce_prey(num_prey, prey_ages, num_clouds, birth_rate, max_prey):
    p_prey_birth = birth_rate * (max_prey - num_prey)
    for cloud in range(num_clouds):
        if np.random.rand(1) < p_prey_birth:
            num_prey += 1
            prey_ages.append(0)

    return num_prey, prey_ages


def age_prey(num_prey, prey_ages, prey_safe_duration, p_prey_death):
    prey_ages = [age + 1 for age in prey_ages]
    to_remove = []
    for i, age in enumerate(prey_ages):
        if age > prey_safe_duration and np.random.rand(1) < p_prey_death:
            to_remove.append(i)
            num_prey -= 1
    for r in reversed(to_remove):
        del prey_ages[r]
    return num_prey, prey_ages


def plot_prey_num_reproduction(birth_rate, num_steps, starting_prey, num_clouds,
                               prey_safe_duration, p_prey_death):
    n_prey_log = []
    n_prey_log.append(starting_prey)
    prey_ages = [0 for i in range(starting_prey)]
    num_prey = starting_prey

    for s in range(num_steps):
        if s == 200:
            num_prey = 5
            prey_ages = prey_ages[:5]
        num_prey, prey_ages = reproduce_prey(num_prey, prey_ages, num_clouds, birth_rate, starting_prey)
        num_prey, prey_ages = age_prey(num_prey, prey_ages, prey_safe_duration, p_prey_death)
        n_prey_log.append(num_prey)

    plt.plot(n_prey_log)
    plt.show()


if __name__ == "__main__":
    plot_prey_num_reproduction(birth_rate=0.001, num_steps=1000, starting_prey=13,
                               num_clouds=16, prey_safe_duration=100, p_prey_death=0.003)











