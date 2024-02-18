import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_circular_current(unit_circle_diameter, current_width, current_strength_variance, max_current_strength,
                            arena_width, arena_height):

    current_strength_variance /= 1
    arena_width /= 1
    arena_height /= 1

    unit_circle_radius = unit_circle_diameter / 2
    circle_width = current_width
    circle_variance = current_strength_variance
    max_current_strength = max_current_strength

    arena_center = np.array([arena_width / 2, arena_height / 2])

    # All coordinates:
    xp, yp = np.arange(arena_width), np.arange(arena_height)
    xy, yp = np.meshgrid(xp, yp)
    xy = np.expand_dims(xy, 2)
    yp = np.expand_dims(yp, 2)
    all_coordinates = np.concatenate((xy, yp), axis=2)
    relative_coordinates = all_coordinates - arena_center  # TO compute coordinates relative to position in center
    distances_from_center = (relative_coordinates[:, :, 0] ** 2 + relative_coordinates[:, :, 1] ** 2) ** 0.5
    distances_from_center = np.expand_dims(distances_from_center, 2)

    xy1 = relative_coordinates[:, :, 0]
    yp1 = relative_coordinates[:, :, 1]
    u = -yp1 / np.sqrt(xy1 ** 2 + yp1 ** 2)
    v = xy1 / np.sqrt(xy1 ** 2 + yp1 ** 2)
    # u, v = np.meshgrid(u, v)
    u = np.expand_dims(u, 2)
    v = np.expand_dims(v, 2)
    vector_field = np.concatenate((u, v), axis=2)

    ### Impose ND structure
    # Compute distance from center at each point
    absolute_distances_from_center = np.absolute(distances_from_center[:, :, 0])
    normalised_distance_from_center = absolute_distances_from_center / np.max(absolute_distances_from_center)
    distance_from_talweg = normalised_distance_from_center - unit_circle_radius
    distance_from_talweg = np.abs(distance_from_talweg)
    distance_from_talweg = np.expand_dims(distance_from_talweg, 2)
    talweg_closeness = (1 - distance_from_talweg) ** 5
    talweg_closeness = (talweg_closeness ** 2) * circle_variance
    current_strength = (talweg_closeness / np.max(talweg_closeness)) * max_current_strength
    current_strength = current_strength[:, :, 0]

    # (Distances - optimal_distance). This forms a subtraction matrix which can be related to the variance.
    adjusted_normalised_distance_from_center = normalised_distance_from_center ** 2

    ### Set cutoffs to 0 outside width
    inside_radius2 = (unit_circle_radius - (circle_width / 2)) ** 2
    outside_radius2 = (unit_circle_radius + (circle_width / 2)) ** 2
    inside = inside_radius2 < adjusted_normalised_distance_from_center
    outside = adjusted_normalised_distance_from_center < outside_radius2
    within_current = inside * outside * 1
    current_strength = current_strength * within_current

    # Scale vector field
    current_strength = np.expand_dims(current_strength, 2)
    vector_field = current_strength * vector_field

    # Prevent middle index being Nan, which causes error.
    vector_field[int(arena_width / 2), int(arena_height / 2)] = 0

    current_strength = (vector_field[:, :, 0] ** 2 + vector_field[:, :, 1] ** 2) ** 0.5

    plt.figure(figsize=(10, 10))
    plt.streamplot(xy[:, :, 0], yp[:, :, 0], vector_field[:, :, 0], vector_field[:, :, 1], color=current_strength,
                   density=2, cmap='Blues')
    plt.xlim(0, arena_width)
    plt.ylim(0, arena_height)

    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == "__main__":
    create_circular_current(unit_circle_diameter=0.7, current_width=0.2, current_strength_variance=1,
                            max_current_strength=1, arena_width=3000, arena_height=3000)
