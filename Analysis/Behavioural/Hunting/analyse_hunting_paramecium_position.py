import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps
from Analysis.Behavioural.Hunting.get_hunting_conditions import get_hunting_conditions
from Analysis.Behavioural.Tools.get_fish_prey_incidence import get_fish_prey_incidence_multiple_prey

"""Find the relation, if any, between initiation of hunting sequences and paramecium position within the visual field"""


def show_conditions_vs_paramecium_angular_position(data, figure_name, absolute_angle=True):
    # Get hunting sequences and associated conditions
    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)
    all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps = get_hunting_conditions(data, all_ts)
    hunting_steps_one_hot = np.array([1 if i in hunting_steps else 0 for i in all_steps])
    initiation_steps_one_hot = np.array([1 if i in initiation_steps else 0 for i in all_steps])
    abort_steps_one_hot = np.array([1 if i in abort_steps else 0 for i in all_steps])
    pre_capture_steps_one_hot = np.array([1 if i in pre_capture_steps else 0 for i in all_steps])

    # Get fish prey incidence for all sequences
    prey_positions = data["prey_positions"]
    fish_positions = data["fish_position"]
    fish_orientation = data["fish_angle"]

    # Repeat so is encoded for each prey
    hunting_steps_one_hot = np.repeat(np.expand_dims(hunting_steps_one_hot, 1), prey_positions.shape[1], 1)
    initiation_steps_one_hot = np.repeat(np.expand_dims(initiation_steps_one_hot, 1), prey_positions.shape[1], 1)
    abort_steps_one_hot = np.repeat(np.expand_dims(abort_steps_one_hot, 1), prey_positions.shape[1], 1)

    # Define conditions where prey is Near.
    fish_prey_vectors = prey_positions - np.expand_dims(fish_positions, 1)
    fish_prey_distance = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    prey_near = np.zeros(fish_prey_distance.shape).astype(int)
    prey_near[fish_prey_distance <= 60] = 1
    any_prey_near = np.sum(prey_near, axis=1)
    any_prey_near[any_prey_near > 0] = 1

    # Get fish prey incidence
    fish_prey_incidence = get_fish_prey_incidence_multiple_prey(fish_positions, fish_orientation, prey_positions)
    if absolute_angle:
        fish_prey_incidence = np.absolute(fish_prey_incidence)

    # Get fish prey incidence for prey that are near
    fish_prey_incidence_prey_near = fish_prey_incidence[prey_near == 1]

    # Get fish prey incidence when prey are near, and a hunting sequence is active
    fish_prey_incidence_prey_near_hunting = fish_prey_incidence[(prey_near == 1) * (hunting_steps_one_hot == 1)]

    # Get fish prey incidence when prey are near, and hunting is initiated
    fish_prey_incidence_prey_near_initiation = fish_prey_incidence[(prey_near == 1) * (initiation_steps_one_hot == 1)]

    # Get fish prey incidence when prey are near, and aborts occur.
    fish_prey_incidence_prey_near_abort = fish_prey_incidence[(prey_near == 1) * (abort_steps_one_hot == 1)]

    # Get fish prey incidence just prior to a successful capture  TODO: Only works if its just the caught prey...
    fish_prey_incidence_pre_capture = fish_prey_incidence[(prey_near == 1) * (np.expand_dims(pre_capture_steps_one_hot, 1) == 1)]

    boxes = [fish_prey_incidence.flatten(), fish_prey_incidence_prey_near, fish_prey_incidence_prey_near_hunting,
             fish_prey_incidence_prey_near_initiation, fish_prey_incidence_prey_near_abort,
             fish_prey_incidence_pre_capture.flatten()]

    # Conditional Boxplots
    fig, ax = plt.subplots()
    ax.boxplot(boxes)
    ax.set_xticklabels(["All Steps", "Prey Near", "Hunting", "Initiation", "Aborts", "Pre-Capture"])

    plt.savefig(f"../../../Analysis-Output/Behavioural/Prey-position-boxplot-{figure_name}.jpg")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    data = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    show_conditions_vs_paramecium_angular_position(data, "test")
