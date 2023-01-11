import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data


def get_fish_prey_incidence_multiple_prey(fish_positions, fish_orientation, paramecium_positions):

    fish_orientation_sign = ((fish_orientation >= 0) * 1) + ((fish_orientation < 0) * -1)

    # Remove full orientations (so is between -2pi and 2pi
    fish_orientation %= 2 * np.pi * fish_orientation_sign

    # Convert to positive scale between 0 and 2pi
    fish_orientation[fish_orientation < 0] += 2 * np.pi

    fish_prey_vectors = paramecium_positions - np.expand_dims(fish_positions, 1)

    # Adjust according to quadrents.
    fish_prey_angles = np.arctan(fish_prey_vectors[:, :, 1] / fish_prey_vectors[:, :, 0])

    #   Generates positive angle from left x axis clockwise.
    # UL quadrent
    in_ul_quadrent = (fish_prey_vectors[:, :, 0] < 0) * (fish_prey_vectors[:, :, 1] > 0)
    fish_prey_angles[in_ul_quadrent] += np.pi
    # BR quadrent
    in_br_quadrent = (fish_prey_vectors[:, :, 0] > 0) * (fish_prey_vectors[:, :, 1] < 0)
    fish_prey_angles[in_br_quadrent] += (np.pi * 2)
    # BL quadrent
    in_bl_quadrent = (fish_prey_vectors[:, :, 0] < 0) * (fish_prey_vectors[:, :, 1] < 0)
    fish_prey_angles[in_bl_quadrent] += np.pi

    # Angle ends up being between 0 and 2pi as clockwise from right x-axis. Same frame as fish angle:
    fish_prey_incidence = np.expand_dims(fish_orientation, 1) - fish_prey_angles

    fish_prey_incidence[fish_prey_incidence > np.pi] %= np.pi
    fish_prey_incidence[fish_prey_incidence < -np.pi] %= -np.pi

    return fish_prey_incidence


def get_fish_prey_incidence(fish_positions, fish_orientation, paramecium_position):
    #
    # fish_orientation_sign = ((fish_orientation >= 0) * 1) + ((fish_orientation < 0) * -1)
    # fish_orientation %= 2 * np.pi * fish_orientation_sign
    #
    # fish_prey_vectors = paramecium_position - np.expand_dims(fish_positions, 1)
    #
    # # Adjust according to quadrents.
    # fish_prey_angles = np.arctan(fish_prey_vectors[:, 1] / fish_prey_vectors[:, 0])
    #
    # #   Generates positive angle from left x axis clockwise.
    # # UL quadrent
    # in_ul_quadrent = (fish_prey_vectors[:, 0] < 0) * (fish_prey_vectors[:, 1] > 0)
    # fish_prey_angles[in_ul_quadrent] += np.pi
    # # BR quadrent
    # in_br_quadrent = (fish_prey_vectors[:, 0] > 0) * (fish_prey_vectors[:, 1] < 0)
    # fish_prey_angles[in_br_quadrent] += (np.pi * 2)
    # # BL quadrent
    # in_bl_quadrent = (fish_prey_vectors[:, 0] < 0) * (fish_prey_vectors[:, 1] < 0)
    # fish_prey_angles[in_bl_quadrent] += np.pi
    # # Angle ends up being between 0 and 2pi as clockwise from right x axis. Same frame as fish angle:
    #
    # fish_prey_incidence = np.absolute(np.expand_dims(fish_orientation, 1) - fish_prey_angles)
    # fish_prey_incidence[fish_prey_incidence > np.pi] -= 2 * np.pi
    #
    # return fish_prey_incidence

    fish_orientation_sign = ((fish_orientation >= 0) * 1) + ((fish_orientation < 0) * -1)

    # Remove full orientations (so is between -2pi and 2pi
    fish_orientation %= 2 * np.pi * fish_orientation_sign

    # Convert to positive scale between 0 and 2pi
    fish_orientation[fish_orientation < 0] += 2 * np.pi

    fish_prey_vectors = paramecium_position - fish_positions

    # Adjust according to quadrents.
    fish_prey_angles = np.arctan(fish_prey_vectors[:, 1] / fish_prey_vectors[:, 0])

    #   Generates positive angle from left x axis clockwise.
    # UL quadrent
    in_ul_quadrent = (fish_prey_vectors[:, 0] < 0) * (fish_prey_vectors[:, 1] > 0)
    fish_prey_angles[in_ul_quadrent] += np.pi
    # BR quadrent
    in_br_quadrent = (fish_prey_vectors[:, 0] > 0) * (fish_prey_vectors[:, 1] < 0)
    fish_prey_angles[in_br_quadrent] += (np.pi * 2)
    # BL quadrent
    in_bl_quadrent = (fish_prey_vectors[:, 0] < 0) * (fish_prey_vectors[:, 1] < 0)
    fish_prey_angles[in_bl_quadrent] += np.pi

    # Angle ends up being between 0 and 2pi as clockwise from right x-axis. Same frame as fish angle:
    fish_prey_incidence = fish_orientation - fish_prey_angles

    fish_prey_incidence[fish_prey_incidence > np.pi] %= np.pi
    fish_prey_incidence[fish_prey_incidence < -np.pi] %= -np.pi

    return fish_prey_incidence


if __name__ == "__main__":
    data = load_data("dqn_beta-1", "Behavioural-Data-Free", "Naturalistic-5")
    prey_positions = data["prey_positions"]
    fish_positions = data["fish_position"]
    fish_orientation = data["fish_angle"]

    fpi = get_fish_prey_incidence_multiple_prey(fish_positions, fish_orientation, prey_positions)



