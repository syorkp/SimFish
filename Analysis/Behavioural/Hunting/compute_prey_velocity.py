import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.convert_prey_position_data_format import convert_prey_position_data


def compute_prey_velocity(prey_positions, fish_positions=None, egocentric=False):
    """If specified egocentric, then computes velocity relative to fish positions"""
    if egocentric:
        prey_positions = np.expand_dims(fish_positions, 1) - prey_positions

    prey_velocity = prey_positions[1:] - prey_positions[:-1]

    # Compute speed
    prey_speed = (prey_velocity[:, :, 0] ** 2 + prey_velocity[:, :, 1] ** 2) ** 0.5

    # Remove those that are too large (due to prey death/reproduction)
    if egocentric:
        to_zero = (prey_speed > 100)
    else:
        to_zero = (prey_speed > 100)

    #
    prey_velocity[to_zero, :] = 0
    prey_speed[to_zero] = 0

    return prey_velocity, prey_speed


if __name__ == "__main__":
    data = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    prey_positions = convert_prey_position_data(data["prey_positions"][:500])
    vel, speed = compute_prey_velocity(prey_positions, egocentric=True, fish_positions=data["fish_position"][:500])
