import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data


def compute_prey_velocity(prey_positions, fish_positions=None, egocentric=False):
    prey_velocity = prey_positions[1:] - prey_positions[:-1]
    # Remove those that are too large (due to prey death/reproduction)

    # Compute speed
    prey_speed = (prey_velocity[:, :, 0] ** 2 + prey_velocity[:, :, 1] ** 2) ** 0.5
    x = True


def convert_prey_position_data(prey_positions_compiled):
    prey_positions_buffer = []

    for prey_positions in prey_positions_compiled:
        prey_positions_reduced = prey_positions[prey_positions[:, 0] < 4000, :]
        prey_positions_buffer.append(prey_positions_reduced)

    # For each step, shift the array of positions across until aligned (should be the min difference with above
    # values).
    num_steps = len(prey_positions_buffer)
    num_prey_init = len(prey_positions_buffer[0])
    overly_large_position_array = np.ones((num_steps, num_prey_init * 100, 2)) * 10000
    min_index = 0
    total_prey_existing = num_prey_init

    for i, prey_position_slice in enumerate(prey_positions_buffer):
        # Ensure one of the arrays is available to accept a new prey.
        overly_large_position_array[i:, total_prey_existing:total_prey_existing+4] = 1000

        if i == 0:
            overly_large_position_array[i, :num_prey_init] = np.array(prey_positions_buffer[i])
        else:
            num_prey = len(prey_positions_buffer[i])
            num_prey_previous = len(overly_large_position_array[i - 1])

            prey_position_slice_expanded = np.repeat(np.expand_dims(prey_position_slice, 1), num_prey_previous, 1)
            prey_position_slice_previous_expanded = np.repeat(np.expand_dims(overly_large_position_array[i - 1], 0),
                                                              num_prey, 0)

            prey_positions_differences = prey_position_slice_expanded - prey_position_slice_previous_expanded
            prey_positions_differences_total = (prey_positions_differences[:, :, 0] ** 2 +
                                                prey_positions_differences[:, :, 1] ** 2) ** 0.5

            forbidden_index = 0

            for prey in range(prey_positions_differences_total.shape[0]):
                differences_to_large_array = prey_positions_differences_total[prey]
                differences_to_large_array[:max([min_index, forbidden_index])] *= 1000
                order_of_size = np.argsort(differences_to_large_array)
                forbidden_index = order_of_size[0]
                if forbidden_index >= total_prey_existing-1:
                    total_prey_existing += 1
                overly_large_position_array[i, forbidden_index] = prey_position_slice[prey]
                forbidden_index += 1

    # Remove columns with only [1000., 1000] or [10000, 10000] (or just both).
    just_1000 = np.sum(((overly_large_position_array[:, :, 0] == 1000.) * (overly_large_position_array[:, :, 1] == 1000.)), axis=0)
    just_10000 = np.sum(((overly_large_position_array[:, :, 0] == 10000.) * (overly_large_position_array[:, :, 1] == 10000.)), axis=0)

    whole_just_1000 = (just_1000 == num_steps) * 1
    whole_just_10000 = (just_10000 == num_steps) * 1
    only_both = ((just_1000 + just_10000) == num_steps) * 1

    to_delete = whole_just_1000 + whole_just_10000 + only_both
    to_delete = [i for i, d in enumerate(to_delete) if d > 0]

    new_prey_position_array = np.delete(overly_large_position_array, to_delete, axis=1)
    new_prey_position_array[new_prey_position_array == 1000.] = 10000.
    return new_prey_position_array


if __name__ == "__main__":
    data = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")
    prey_positions = convert_prey_position_data(data["prey_positions"][:500])
    compute_prey_velocity(prey_positions)
