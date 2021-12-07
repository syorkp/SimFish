import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import skimage.draw as draw
from skimage import io

from Tools.ray_cast import rays


class NewDrawingBoard:

    def __init__(self, width, height, decay_rate, photoreceptor_rf_size, using_gpu):

        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.photoreceptor_rf_size = photoreceptor_rf_size
        self.db = None
        self.erase()

        # Set of coordinates
        # xp, yp = np.meshgrid(range(self.width), range(self.height))
        # self.coordinates = np.concatenate((xp, yp), 3)

        # xp, yp = cp.meshgrid(cp.arange(self.width), cp.arange(self.height))
        # self.coordinates = cp.concatenate((xp, yp), 2)

        # self.scatter = cp.vectorize(lambda i, j, x, y: np.exp(-self.decay_rate * (((x - i) ** 2 + (y - j) ** 2) ** 0.5)))
        if using_gpu:
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

        self.xp, self.yp = self.chosen_math_library.arange(self.width), self.chosen_math_library.arange(self.height)

    def scatter(self, i, j, x, y):
        # scatter = cp.exp(-self.decay_rate * (((x - i) ** 2 + (y - j) ** 2) ** 0.5))
        # return cp.expand_dims(scatter, 2)
        positional_mask = (((x - i) ** 2 + (y - j) ** 2) ** 0.5)
        desired_scatter = self.chosen_math_library.exp(-self.decay_rate * positional_mask)
        implicit_scatter = self.chosen_math_library.sin(self.photoreceptor_rf_size) * positional_mask
        implicit_scatter[implicit_scatter < 1] = 1
        adjusted_scatter = desired_scatter * implicit_scatter
        adjusted_scatter = self.chosen_math_library.expand_dims(adjusted_scatter, 2)
        return adjusted_scatter

    @staticmethod
    def apply_mask(board, mask):
        # TODO: speed up
        # new_board = np.zeros(board.shape)
        # for channel in range(board.shape[-1]):
        #     new_board[:, :, channel] = np.multiply(board[:, :, channel], mask)
        # return new_board
        mask = np.expand_dims(mask, 2)
        return board * mask

    def decay(self, fish_position):
        return np.exp(-self.decay_rate * (((fish_position[0] - i) ** 2 + (fish_position[1] - j) ** 2) ** 0.5))

    def create_scatter_mask(self, fish_position):
        """NO LONGER USED.
        Creates the scatter mask according to the equation: I(d)=e^(-decay_rate*d), where d is distance from fish,
        computed here for all coordinates."""
        mask = np.fromfunction(
            lambda i, j: np.exp(-self.decay_rate * (((fish_position[0] - i) ** 2 + (fish_position[1] - j) ** 2) ** 0.5)),
            (self.width, self.height,),
            dtype=float)
        mask = np.expand_dims(mask, 2)
        return mask

    def create_obstruction_mask(self, fish_position, prey_locations, predator_locations):
        n_objects_to_check = len(prey_locations)
        prey_half_size = 2

        fish_position = np.array(fish_position)
        prey_locations = np.array(prey_locations)

        relative_positions = prey_locations-fish_position

        prey_distances = (relative_positions[:, 0] ** 2 + relative_positions[:, 1] ** 2) ** 0.5
        prey_half_angular_size = np.arctan(prey_half_size/prey_distances)

        object_angles = np.arctan(relative_positions[:, 1]/relative_positions[:, 0])
        object_angles = np.expand_dims(object_angles, 1)
        object_angles = np.repeat(object_angles, 2, 1)

        rf_offsets = np.expand_dims(prey_half_angular_size, 1)
        rf_offsets = np.repeat(rf_offsets, 2, 1)
        rf_offsets = rf_offsets * np.array([-1, 1])
        object_extremities = object_angles + rf_offsets

        below_range = (object_extremities < 0) * np.pi * 2
        object_extremities = object_extremities + below_range
        above_range = (object_extremities > np.pi * 2) * -np.pi*2
        object_extremities = object_extremities + above_range

        # Compute m using tan (N_obj x 2)
        m = np.tan(object_extremities)

        # Compute c (N_obj x 2)
        c = -m * fish_position[0]
        c = c + fish_position[1]

        # Compute components of intersections (N_obj x 2 x 4)
        c = np.expand_dims(c, 2)
        c = np.repeat(c, 4, 2)

        multiplication_matrix_unit = np.array([-1, 1, -1, 1])
        multiplication_matrix = np.tile(multiplication_matrix_unit, (n_objects_to_check, 2, 1))

        addition_matrix_unit = np.array([0, 0, self.height, self.width])
        addition_matrix = np.tile(addition_matrix_unit, (n_objects_to_check, 2, 1))

        mul1 = np.array([0, 0, 0, 1])
        mul1_full = np.tile(mul1, (n_objects_to_check, 2, 1))
        m_mul = np.expand_dims(m, 2)
        full_m = np.repeat(m_mul, 4, 2)
        m_mul = full_m * mul1_full
        m_mul[:, :, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c * multiplication_matrix) + addition_matrix)/division_matrix

        mul_for_hypothetical = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        mul_for_hypothetical = np.tile(mul_for_hypothetical, (n_objects_to_check, 2, 1, 1))
        add_for_hypothetical = np.array([[0, 0], [0, 0], [0, self.width], [self.height, 0]])
        add_for_hypothetical = np.tile(add_for_hypothetical, (n_objects_to_check, 2, 1, 1))

        intersection_coordinates = np.expand_dims(intersection_components, 3)
        intersection_coordinates = np.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (N_obj x 2 x 2 x 2)
        conditional_tiled = np.array([self.width, self.height, self.width, self.height])
        conditional_tiled = np.tile(conditional_tiled, (n_objects_to_check, 2, 1))
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = np.reshape(valid_intersection_coordinates, (n_objects_to_check, 2, 2, 2))

        # Get intersections (N_obj x 2)
        eye_position = np.array(fish_position)
        possible_vectors = valid_intersection_coordinates - eye_position
        angles = np.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        below_range = (angles < 0) * np.pi * 2
        angles = angles + below_range
        above_range = (angles > np.pi * 2) * -np.pi*2
        angles = angles + above_range

        angles = np.round(angles, 2)
        channel_angles_surrounding = np.round(object_extremities, 2)

        channel_angles_surrounding = np.expand_dims(channel_angles_surrounding, 2)
        channel_angles_surrounding = np.repeat(channel_angles_surrounding, 2, 2)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = np.reshape(selected_intersections, (n_objects_to_check, 2, 2))

        # Finding coordinates of object extremities.
        proj_vector = selected_intersections - fish_position
        proj_distance = (proj_vector[:, :, 0] ** 2 + proj_vector[:, :, 1] ** 2) ** 0.5  # Only really need to do for one as is same distance along.
        # norm_proj_vector = proj_vector/proj_distance
        distance_along = (prey_distances ** 2 + prey_half_size ** 2) ** 0.5
        distance_along = np.expand_dims(distance_along, 1)
        distance_along = np.repeat(distance_along, 2, 1)

        fraction_along = distance_along/proj_distance
        fraction_along = np.expand_dims(fraction_along, 2)
        fraction_along = np.repeat(fraction_along, 2, 2)

        points_on_prey = proj_vector * fraction_along
        points_on_prey = fish_position + points_on_prey

        coordinates_of_occlusion = np.concatenate((selected_intersections, points_on_prey), 1)

        # Getting corner vertices
        x_vertices = coordinates_of_occlusion[:, :, 0]
        y_vertices = coordinates_of_occlusion[:, :, 1]

        min_x = np.min(x_vertices, 1)
        min_y = np.min(y_vertices, 1)
        max_x = np.max(x_vertices, 1)
        max_y = np.max(y_vertices, 1)

        x_wall_left = (min_x == 0) * 1
        x_wall_right = (max_x == self.width) * 1
        y_wall_top = (min_y == 0) * 1
        y_wall_bottom = (max_y == self.width) * 1

        top_left = x_wall_left * y_wall_top
        bottom_left = x_wall_left * y_wall_bottom
        top_right = x_wall_right * y_wall_top
        bottom_right = x_wall_right * y_wall_bottom
        # TODO: Can then set values within these corners to 0 (can simply set values < other vertices and > other vertices to 0).

        # TODO: might be faster just to do lines again between the two angle extremeties...

        # Computing points in segments
        x = True

    def create_obstruction_mask_lines(self, fish_position, prey_locations, predator_locations):
        n = 20
        # TODO: Compute n * prey num once.

        n_objects_to_check = len(prey_locations)
        prey_half_size = 2

        fish_position = self.chosen_math_library.array(fish_position)
        prey_locations = self.chosen_math_library.array(prey_locations)

        relative_positions = prey_locations-fish_position

        prey_distances = (relative_positions[:, 0] ** 2 + relative_positions[:, 1] ** 2) ** 0.5
        prey_half_angular_size = self.chosen_math_library.arctan(prey_half_size/prey_distances)

        object_angles = self.chosen_math_library.arctan(relative_positions[:, 1]/relative_positions[:, 0])
        object_angles = self.chosen_math_library.expand_dims(object_angles, 1)
        object_angles = self.chosen_math_library.repeat(object_angles, 2, 1)

        rf_offsets = self.chosen_math_library.expand_dims(prey_half_angular_size, 1)
        rf_offsets = self.chosen_math_library.repeat(rf_offsets, 2, 1)
        rf_offsets = rf_offsets * self.chosen_math_library.array([-1, 1])
        object_extremities = object_angles + rf_offsets

        interpolated_line_angles = self.chosen_math_library.linspace(object_extremities[:, 0], object_extremities[:, 1], n).flatten()

        below_range = (interpolated_line_angles < 0) * self.chosen_math_library.pi * 2
        interpolated_line_angles = interpolated_line_angles + below_range
        above_range = (interpolated_line_angles > self.chosen_math_library.pi * 2) * -self.chosen_math_library.pi*2
        interpolated_line_angles = interpolated_line_angles + above_range

        # Compute m using tan (N_obj x n)
        m = self.chosen_math_library.tan(interpolated_line_angles)

        # Compute c (N_obj*n)
        c = -m * fish_position[0]
        c = c + fish_position[1]

        # Compute components of intersections (N_obj*n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 1)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 1)

        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit, (n_objects_to_check*n, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.height-1, self.width-1])
        addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (n_objects_to_check*n, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        mul1_full = self.chosen_math_library.tile(mul1, (n_objects_to_check*n, 1))
        m_mul = self.chosen_math_library.expand_dims(m, 1)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 1)
        m_mul = full_m * mul1_full
        m_mul[:, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, 1] = 1
        division_matrix[:, 3] = 1

        intersection_components = ((c_exp * multiplication_matrix) + addition_matrix)/division_matrix

        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (n_objects_to_check*n, 1, 1))
        add_for_hypothetical = self.chosen_math_library.array([[0, 0], [0, 0], [0, self.width-1], [self.height-1, 0]])
        add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (n_objects_to_check*n, 1, 1))

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 2)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 2)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (N_obj n 2 x 2 x 2)
        conditional_tiled = self.chosen_math_library.array([self.width-1, self.height-1, self.width-1, self.height-1])
        conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (n_objects_to_check*n, 1))
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]

        # Get intersections (N_obj x 2)
        eye_position = self.chosen_math_library.array(fish_position)
        possible_vectors = valid_intersection_coordinates - eye_position
        angles = self.chosen_math_library.arctan2(possible_vectors[:, 1], possible_vectors[:, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        below_range = (angles < 0) * self.chosen_math_library.pi * 2
        angles = angles + below_range
        above_range = (angles > self.chosen_math_library.pi * 2) * -self.chosen_math_library.pi*2
        angles = angles + above_range

        angles = self.chosen_math_library.round(angles, 2)
        channel_angles_surrounding = self.chosen_math_library.round(interpolated_line_angles, 2)

        channel_angles_surrounding = self.chosen_math_library.expand_dims(channel_angles_surrounding, 1)
        channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, 2, 1).flatten()

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]

        # TODO: replace eye position with computation of vertices
        # Finding coordinates of object extremities.
        proj_vector = selected_intersections - fish_position
        proj_distance = (proj_vector[:, 0] ** 2 + proj_vector[:, 1] ** 2) ** 0.5  # Only really need to do for one as is same distance along.
        # norm_proj_vector = proj_vector/proj_distance
        distance_along = (prey_distances ** 2 + prey_half_size ** 2) ** 0.5
        distance_along = self.chosen_math_library.expand_dims(distance_along, 1)
        distance_along = self.chosen_math_library.repeat(distance_along, n, 1)
        distance_along = self.chosen_math_library.swapaxes(distance_along, 0, 1).flatten()
        distance_along = distance_along + prey_half_size

        fraction_along = distance_along/proj_distance
        fraction_along = self.chosen_math_library.expand_dims(fraction_along, 1)
        fraction_along = self.chosen_math_library.repeat(fraction_along, 2, 1)

        points_on_prey = proj_vector * fraction_along
        points_on_prey = fish_position + points_on_prey
        points_on_prey = self.chosen_math_library.expand_dims(points_on_prey, 1)

        selected_intersections = self.chosen_math_library.reshape(selected_intersections, (n_objects_to_check*n, 1, 2))

        vertices = self.chosen_math_library.concatenate((selected_intersections, points_on_prey), 1)
        vertices_xvals = vertices[:, :, 0]
        vertices_yvals = vertices[:, :, 1]

        # INTERPOLATION
        # TODO: Probably faster way of doing below...
        min_x = self.chosen_math_library.min(vertices_xvals, axis=1)
        max_x = self.chosen_math_library.max(vertices_xvals, axis=1)
        min_y = self.chosen_math_library.min(vertices_yvals, axis=1)
        max_y = self.chosen_math_library.max(vertices_yvals, axis=1)

        # SEGMENT COMPUTATION  # TODO: Make sure this is enough to cover span.
        x_lens = self.chosen_math_library.rint(max_x[0] - min_x[0])
        y_lens = self.chosen_math_library.rint(max_y[0] - min_y[0])

        x_len = self.chosen_math_library.max(x_lens)
        y_len = self.chosen_math_library.max(y_lens)

        x_ranges = self.chosen_math_library.linspace(min_x, max_x, int(x_len))
        y_ranges = self.chosen_math_library.linspace(min_y, max_y, int(y_len))

        y_values = (m * x_ranges) + c
        y_values = self.chosen_math_library.floor(y_values)
        set_1 = self.chosen_math_library.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = self.chosen_math_library.floor(x_values)
        set_2 = self.chosen_math_library.stack((x_values, y_ranges), axis=-1)
        full_set = self.chosen_math_library.vstack((set_1, set_2)).astype(int)

        full_set = full_set.reshape(-1, 2)
        mask = self.chosen_math_library.ones((1500, 1500), dtype=int)

        mask[full_set[:, 0], full_set[:, 1]] = 0
        # plt.imshow(mask)
        # plt.show()
        return mask

    def create_obstruction_mask_lines_np(self, fish_position, prey_locations, predator_locations):
        n = 20
        # TODO: Compute n * prey num once.

        n_objects_to_check = len(prey_locations)
        prey_half_size = 2

        fish_position = np.array(fish_position)
        prey_locations = np.array(prey_locations)

        relative_positions = prey_locations-fish_position

        prey_distances = (relative_positions[:, 0] ** 2 + relative_positions[:, 1] ** 2) ** 0.5
        prey_half_angular_size = np.arctan(prey_half_size/prey_distances)

        object_angles = np.arctan(relative_positions[:, 1]/relative_positions[:, 0])
        object_angles = np.expand_dims(object_angles, 1)
        object_angles = np.repeat(object_angles, 2, 1)

        rf_offsets = np.expand_dims(prey_half_angular_size, 1)
        rf_offsets = np.repeat(rf_offsets, 2, 1)
        rf_offsets = rf_offsets * np.array([-1, 1])
        object_extremities = object_angles + rf_offsets

        interpolated_line_angles = np.linspace(object_extremities[:, 0], object_extremities[:, 1], n).flatten()

        below_range = (interpolated_line_angles < 0) * np.pi * 2
        interpolated_line_angles = interpolated_line_angles + below_range
        above_range = (interpolated_line_angles > np.pi * 2) * - np.pi*2
        interpolated_line_angles = interpolated_line_angles + above_range

        # Compute m using tan (N_obj x n)
        m = np.tan(interpolated_line_angles)

        # Compute c (N_obj*n)
        c = -m * fish_position[0]
        c = c + fish_position[1]

        # Compute components of intersections (N_obj*n x 4)
        c_exp = np.expand_dims(c, 1)
        c_exp = np.repeat(c_exp, 4, 1)

        multiplication_matrix_unit = np.array([-1, 1, -1, 1])
        multiplication_matrix = np.tile(multiplication_matrix_unit, (n_objects_to_check*n, 1))

        addition_matrix_unit = np.array([0, 0, self.height-1, self.width-1])
        addition_matrix = np.tile(addition_matrix_unit, (n_objects_to_check*n, 1))

        mul1 = np.array([0, 0, 0, 1])
        mul1_full = np.tile(mul1, (n_objects_to_check*n, 1))
        m_mul = np.expand_dims(m, 1)
        full_m = np.repeat(m_mul, 4, 1)
        m_mul = full_m * mul1_full
        m_mul[:, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, 1] = 1
        division_matrix[:, 3] = 1

        intersection_components = ((c_exp * multiplication_matrix) + addition_matrix)/division_matrix

        mul_for_hypothetical = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        mul_for_hypothetical = np.tile(mul_for_hypothetical, (n_objects_to_check*n, 1, 1))
        add_for_hypothetical = np.array([[0, 0], [0, 0], [0, self.width-1], [self.height-1, 0]])
        add_for_hypothetical = np.tile(add_for_hypothetical, (n_objects_to_check*n, 1, 1))

        intersection_coordinates = np.expand_dims(intersection_components, 2)
        intersection_coordinates = np.repeat(intersection_coordinates, 2, 2)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (N_obj n 2 x 2 x 2)
        conditional_tiled = np.array([self.width-1, self.height-1, self.width-1, self.height-1])
        conditional_tiled = np.tile(conditional_tiled, (n_objects_to_check*n, 1))
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]

        # Get intersections (N_obj x 2)
        eye_position = np.array(fish_position)
        possible_vectors = valid_intersection_coordinates - eye_position
        angles = np.arctan2(possible_vectors[:, 1], possible_vectors[:, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        below_range = (angles < 0) * np.pi * 2
        angles = angles + below_range
        above_range = (angles > np.pi * 2) * - np.pi*2
        angles = angles + above_range

        angles = np.round(angles, 2)
        channel_angles_surrounding = np.round(interpolated_line_angles, 2)
        channel_angles_surrounding = np.expand_dims(channel_angles_surrounding, 1)
        channel_angles_surrounding = np.repeat(channel_angles_surrounding, 2, 1).flatten()

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]

        # TODO: replace eye position with computation of vertices
        # Finding coordinates of object extremities.
        proj_vector = selected_intersections - fish_position
        proj_distance = (proj_vector[:, 0] ** 2 + proj_vector[:, 1] ** 2) ** 0.5  # Only really need to do for one as is same distance along.
        # norm_proj_vector = proj_vector/proj_distance
        distance_along = (prey_distances ** 2 + prey_half_size ** 2) ** 0.5
        distance_along = np.expand_dims(distance_along, 1)
        distance_along = np.repeat(distance_along, n, 1)
        distance_along = np.swapaxes(distance_along, 0, 1).flatten()
        distance_along = distance_along + prey_half_size

        fraction_along = distance_along/proj_distance
        fraction_along = np.expand_dims(fraction_along, 1)
        fraction_along = np.repeat(fraction_along, 2, 1)

        points_on_prey = proj_vector * fraction_along
        points_on_prey = fish_position + points_on_prey
        points_on_prey = np.expand_dims(points_on_prey, 1)

        selected_intersections = np.reshape(selected_intersections, (n_objects_to_check*n, 1, 2))

        vertices = np.concatenate((selected_intersections, points_on_prey), 1)
        vertices_xvals = vertices[:, :, 0]
        vertices_yvals = vertices[:, :, 1]

        # INTERPOLATION
        # TODO: Probably faster way of doing below...
        min_x = np.min(vertices_xvals, axis=1)
        max_x = np.max(vertices_xvals, axis=1)
        min_y = np.min(vertices_yvals, axis=1)
        max_y = np.max(vertices_yvals, axis=1)

        # SEGMENT COMPUTATION  # TODO: Make sure this is enough to cover span.
        x_lens = np.rint(max_x[0] - min_x[0])
        y_lens = np.rint(max_y[0] - min_y[0])

        x_len = np.max(x_lens)
        y_len = np.max(y_lens)

        x_ranges = np.linspace(min_x, max_x, int(x_len))
        y_ranges = np.linspace(min_y, max_y, int(y_len))

        y_values = (m * x_ranges) + c
        y_values = np.floor(y_values)
        set_1 = np.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = np.floor(x_values)
        set_2 = np.stack((x_values, y_ranges), axis=-1)
        full_set = np.vstack((set_1, set_2)).astype(int)

        full_set = full_set.reshape(-1, 2)
        mask = np.ones((1500, 1500), dtype=int)

        mask[full_set[:, 0], full_set[:, 1]] = 0
        # plt.imshow(mask)
        # plt.show()
        return mask

    def create_luminance_mask(self):
        # TODO: implement.
        return np.ones((self.width, self.height, 1))

    def get_masked_pixels_cupy(self, fish_position, prey_locations, predator_locations=[]):
        A = self.chosen_math_library.array(self.db)
        L = self.chosen_math_library.ones((self.width, self.height, 1))
        # O = self.create_obstruction_mask_lines(fish_position, prey_locations, predator_locations)
        O = cp.array(self.create_obstruction_mask_lines_np(fish_position, prey_locations, predator_locations))
        O = self.chosen_math_library.expand_dims(O, 2)
        S = self.scatter(self.xp[:, None], self.yp[None, :], fish_position[0], fish_position[1])
        return A * L * O * S

    def get_masked_pixels(self, fish_position):
        A = self.db
        L = self.create_luminance_mask()
        O = self.create_obstruction_mask(fish_position)
        S = self.create_scatter_mask(fish_position)
        return A * L * O * S
        # masked_arena = self.apply_mask(self.apply_mask(self.apply_mask(A, L), O), S)
        # return masked_arena

    def erase(self, bkg=0):
        if bkg == 0:
            self.db = np.zeros((self.height, self.width, 3), dtype=np.double)
        else:
            self.db = np.ones((self.height, self.width, 3), dtype=np.double) * bkg
        self.draw_walls()

    def draw_walls(self):
        self.db[0:2, :] = [1, 0, 0]
        self.db[self.width-1, :] = [1, 0, 0]
        self.db[:, 0] = [1, 0, 0]
        self.db[:, self.height-1] = [1, 0, 0]

    def apply_light(self, dark_col, dark_gain, light_gain):
        self.db[:, :dark_col] *= dark_gain
        self.db[:, dark_col:] *= light_gain

    def circle(self, center, rad, color):
        rr, cc = draw.circle(center[1], center[0], rad, self.db.shape)
        self.db[rr, cc, :] = color

    def tail(self, head, left, right, tip, color):
        tail_coordinates = np.array((head, left, tip, right))
        rr, cc = draw.polygon(tail_coordinates[:, 1], tail_coordinates[:, 0], self.db.shape)
        self.db[rr, cc, :] = color

    def fish_shape(self, mouth_centre, mouth_rad, head_rad, tail_length, mouth_colour, body_colour, angle):
        offset = np.pi / 2
        angle += offset
        angle = -angle
        self.circle(mouth_centre, mouth_rad, mouth_colour)  # For the mouth.
        dx1, dy1 = head_rad * np.sin(angle), head_rad * np.cos(angle)
        head_centre = (mouth_centre[0] + dx1,
                       mouth_centre[1] + dy1)
        self.circle(head_centre, head_rad, body_colour)
        dx2, dy2 = -1 * dy1, dx1
        left_flank = (head_centre[0] + dx2,
                      head_centre[1] + dy2)
        right_flank = (head_centre[0] - dx2,
                       head_centre[1] - dy2)
        tip = (mouth_centre[0] + (tail_length + head_rad) * np.sin(angle),
               mouth_centre[1] + (tail_length + head_rad) * np.cos(angle))
        self.tail(head_centre, left_flank, right_flank, tip, body_colour)

    def create_screen(self, fish_position, distance, colour):
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 10)
        self.db[rr, cc, :] = colour
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 9)
        self.db[rr, cc, :] = colour
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 8)
        self.db[rr, cc, :] = colour

    def vegetation(self, vertex, edge_size, color):
        coordinates = np.array(((vertex[1], vertex[0]),
                                (vertex[1], vertex[0] + edge_size),
                                (vertex[1] + edge_size / 2, vertex[0] + edge_size - edge_size / 3),
                                (vertex[1] + edge_size, vertex[0] + edge_size),
                                (vertex[1] + edge_size, vertex[0]),
                                (vertex[1] + edge_size / 2, vertex[0] + edge_size / 3)))

        rr, cc = draw.polygon(coordinates[:, 0], coordinates[:, 1], self.db.shape)
        self.db[rr, cc, :] = color

    @staticmethod
    def multi_circles(cx, cy, rad):
        rr, cc = draw.circle(0, 0, rad)
        rrs = np.tile(rr, (len(cy), 1)) + np.tile(np.reshape(cy, (len(cy), 1)), (1, len(rr)))
        ccs = np.tile(cc, (len(cx), 1)) + np.tile(np.reshape(cx, (len(cx), 1)), (1, len(cc)))
        return rrs, ccs

    def line(self, p1, p2, color):
        rr, cc = draw.line(p1[1], p1[0], p2[1], p2[0])
        self.db[rr, cc, :] = color

    def get_size(self):
        return self.width, self.height

    def read_rays(self, xmat, ymat, dark_gain, light_gain, bkg_scatter, dark_col=0):
        res = rays(xmat.astype(np.int), ymat.astype(np.int), self.db, self.height, self.width, dark_gain, light_gain,
                   bkg_scatter, dark_col=dark_col)
        return res

    def read(self, xmat, ymat):
        n_arms = xmat.shape[0]
        res = np.zeros((n_arms, 3))
        for arm in range(n_arms):
            [rr, cc] = draw.line(ymat[arm, 0].astype(int), xmat[arm, 0].astype(int), ymat[arm, 1].astype(int),
                                 xmat[arm, 1].astype(int))
            prfl = self.db[rr, cc]
            # prfl = np.array(profile_line(self.db, (ymat[arm,0], xmat[arm,0]), (ymat[arm,1], xmat[arm,1]), order=0, cval=1.))
            ps = np.sum(prfl, 1)
            if len(np.nonzero(ps)[0]) > 0:
                res[arm, :] = prfl[np.nonzero(ps)[0][0], :]
            else:
                res[arm, :] = [0, 0, 0]

        # xmat_ = np.where((xmat<0) | (xmat>=self.width), 0, xmat)
        # ymat_ = np.where((ymat<0) | (ymat>=self.height), 0, ymat)
        #
        # res = self.db[ymat_, xmat_, :]
        # res[np.where((xmat<0)|xmat>=self.width)|(ymat<0)|(ymat>=self.height), :] = [1, 0, 0]
        return res

    def show(self):
        io.imshow(self.db)
        io.show()


if __name__ == "__main__":
    d = DrawingBoard(500, 500)
    d.circle((100, 200), 100, (1, 0, 0))
    d.line((50, 50), (100, 200), (0, 1, 0))
    d.show()
