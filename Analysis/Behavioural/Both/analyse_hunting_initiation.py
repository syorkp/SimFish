import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps
from Analysis.Behavioural.Both.analyse_hunting import get_paramecium_density

"""
Aim should be to find the parameters that determine success of hunting, hunting sequence initiation, etc. 
"""


def show_initiation_vs_prey_density(data, figure_name):
    """Coloured scatter plot to indicate all prey density values and whether a hunt was initiated at the time"""
    # Get all steps for which there was hunting initiation
    # Get all steps for which there was exiting of a hunting sequence
    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)

    hunting_steps = list(set([a for at in all_ts for a in at]))
    initiation_steps = np.array(list(set([a[0] for a in all_ts])))

    paramecium_density = np.array([get_paramecium_density(data["fish_position"][i], data["prey_positions"][i])
                          for i in range(data["fish_position"].shape[0])])
    color = np.array([1 if i in hunting_steps else 0 for i, p in enumerate(paramecium_density)])

    failed_sequences = []
    for ts in all_ts:
        if data["consumed"][ts[-1]+1]:
            pass
        else:
            failed_sequences.append(ts)

    abort_steps = np.array(list(set([a[-1] for a in all_ts])))

    # Time scatter plot
    plt.scatter([i for i in range(len(paramecium_density))], paramecium_density, c=color)
    plt.savefig(f"../../../Analysis-Output/Behavioural/Initiation_vs_prey_density-{figure_name}.jpg")
    plt.clf()
    plt.close()

    boxes = [paramecium_density[color == 0], paramecium_density[color == 1], paramecium_density[initiation_steps], paramecium_density[abort_steps]]
    # Conditional histograms
    fig, ax = plt.subplots()
    ax.boxplot(boxes)
    ax.set_xticklabels(["Normal", "Hunting", "Initiation", "Aborts"])

    plt.savefig(f"../../../Analysis-Output/Behavioural/Initiation_vs_prey_density-boxplot-{figure_name}.jpg")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    d1 = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    show_initiation_vs_prey_density(d1, "Test")
