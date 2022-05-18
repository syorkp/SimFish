import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data


def prey_position():
    stimulus_data = load_stimulus_data("new_differential_prey_ref-3", f"Prey-Full-Response-Vector", "Prey-Right-10")
    diff = stimulus_data[0]["prey 1"]["Onset"] - stimulus_data[0]["prey 1"]["Pre-onset"]
    steps = [[l["prey 1"]["Onset"], l["prey 1"]["Onset"]+diff] for l in stimulus_data]
    angles = [l["prey 1"]["Angle"] for l in stimulus_data]

    interpolated_steps = [list(range(st[0], st[1])) for st in steps]
    # interpolated_steps = []
    # for i in int_steps:
    #     interpolated_steps += i

    interpolated_angles = []
    for i, a in enumerate(angles):
        if i == len(angles) - 1:
            interpolated_angles.append(list(np.linspace(a, a + (angles[1]-angles[0]), diff)))
        else:
            interpolated_angles.append(list(np.linspace(a, angles[i+1], diff)))

    sns.set()
    for i, step in enumerate(interpolated_steps):
        if i == len(angles) - 1:
            continue
        plt.plot(step, interpolated_angles[i])
    plt.ylabel("Stimulus Angle (pi radians)")
    plt.xlabel("Time (steps)")
    plt.show()


def real_prey_position():
    data = load_data("new_even_prey_ref-3", f"Behavioural-Data-Free", "Prey-1")
    sns.set()
    chosen_prey_position = [step[5] for step in data["prey_positions"]][:100]
    plt.plot([i[0] for i in chosen_prey_position], [i[1] for i in chosen_prey_position])
    plt.show()
    x = True

real_prey_position()

