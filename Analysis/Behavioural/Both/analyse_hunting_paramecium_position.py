import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps
from Analysis.Behavioural.Both.get_hunting_conditions import get_hunting_conditions
from Analysis.Behavioural.Tools.get_fish_prey_incidence import get_fish_prey_incidence_multiple_prey

"""Find the relation, if any, between initiation of hunting sequences and paramecium position within the visual field"""


def show_conditions_vs_paramecium_angular_position(data, figure_name):
    # Get hunting sequences and associated conditions
    all_seq, all_ts = get_hunting_sequences_timestamps(data, False)
    all_steps, hunting_steps, initiation_steps, abort_steps = get_hunting_conditions(data, all_ts)

    # Get fish prey incidence for all sequences
    prey_positions = data["prey_positions"]
    fish_positions = data["fish_position"]
    fish_orientation = data["fish_angle"]

    # Define conditions where prey is Near.
    fish_prey_vectors = prey_positions - np.expand_dims(fish_positions, 1)
    fish_prey_distance = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    prey_near = np.zeros(fish_prey_distance.shape).astype(int)
    prey_near[fish_prey_distance <= 60] = 1

    # Get fish prey incidence
    fish_prey_incidence = get_fish_prey_incidence_multiple_prey(fish_positions, fish_orientation, prey_positions)

    # Get fish prey incidence for prey that are near
    fish_prey_incidence_prey_near = fish_prey_incidence[prey_near == 1]

    # TODO: Get fish prey incidence when prey are near, and a hunting sequence is active

    fish_prey_incidence_prey_near_hunting = fish_prey_incidence[]

    # Get fish prey incidence when prey are near, and each of the above conditions

    x = True


if __name__ == "__main__":
    data = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    show_conditions_vs_paramecium_angular_position(data, "test")
