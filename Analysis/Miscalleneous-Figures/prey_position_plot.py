import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Analysis.load_stimuli_data import load_stimulus_data


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


prey_position()

