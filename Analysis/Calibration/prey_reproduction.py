import numpy as np
import matplotlib.pyplot as plt
import random

from Analysis.load_data import load_data


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


def fish_caught_prey(num_prey, prey_ages):
    prey_to_die = random.randint(0, num_prey-1)
    num_prey -= 1
    del prey_ages[prey_to_die]
    return num_prey, prey_ages

def plot_prey_num_reproduction(birth_rate, num_steps, starting_prey, num_clouds,
                               prey_safe_duration, p_prey_death, consumption_events=None):
    n_prey_log = []
    n_prey_log.append(starting_prey)
    prey_ages = [0 for i in range(starting_prey)]
    num_prey = starting_prey

    for s in range(num_steps):
        num_prey, prey_ages = reproduce_prey(num_prey, prey_ages, num_clouds, birth_rate, starting_prey)
        num_prey, prey_ages = age_prey(num_prey, prey_ages, prey_safe_duration, p_prey_death)
        if consumption_events is not None:
            if consumption_events[s]:
                num_prey, prey_ages = fish_caught_prey(num_prey, prey_ages)
        n_prey_log.append(num_prey)

    plt.plot(n_prey_log)
    plt.show()


if __name__ == "__main__":
    consumption_events = load_data("dqn_scaffold_26-2", "Behavioural-Data-Free", "Naturalistic-1")["consumed"]
    # consumption_events[:] = False
    consumption_events = np.concatenate((consumption_events, consumption_events))

    plot_prey_num_reproduction(birth_rate=0.002, num_steps=consumption_events.shape[0], starting_prey=100,
                               num_clouds=16, prey_safe_duration=100, p_prey_death=0.001,
                               consumption_events=consumption_events)











