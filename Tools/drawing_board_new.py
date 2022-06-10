import numpy as np
import cupy as cp
import math
import matplotlib.pyplot as plt
from time import time

import skimage.draw as draw
from skimage import io


class NewDrawingBoard:

    def __init__(self, width, height, decay_rate, photoreceptor_rf_size, using_gpu, visualise_mask, prey_size=4,
                 predator_size=100, visible_scatter=0.3, background_grating_frequency=50, dark_light_ratio=0.0,
                 dark_gain=0.01, light_gain=1.0, red_occlusion_gain=1.0, uv_occlusion_gain=1.0,
                 red2_occlusion_gain=1.0, light_gradient=0):

        self.using_gpu = using_gpu

        if using_gpu:
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.light_gain = light_gain
        self.light_gradient = light_gradient
        self.photoreceptor_rf_size = photoreceptor_rf_size
        self.db = None
        self.db_visualisation = None
        self.base_db = self.get_base_arena()
        self.base_db_illuminated = self.get_base_arena(visible_scatter)
        self.background_grating = self.get_background_grating(background_grating_frequency)
        self.luminance_mask = self.get_luminance_mask(dark_light_ratio, dark_gain)
        self.erase(visible_scatter)

        self.prey_size = prey_size * 2
        self.prey_radius = prey_size
        self.predator_size = predator_size * 2
        self.predator_radius = predator_size

        self.red_occlusion_gain = red_occlusion_gain
        self.uv_occlusion_gain = uv_occlusion_gain
        self.red2_occlusion_gain = red2_occlusion_gain

        if self.red_occlusion_gain != self.uv_occlusion_gain or self.uv_occlusion_gain != self.red2_occlusion_gain:
            self.differential_occlusion_gain = True
        else:
            self.differential_occlusion_gain = False

        self.xp, self.yp = self.chosen_math_library.arange(self.width), self.chosen_math_library.arange(self.height)

        if self.using_gpu:
            self.max_lines_num = 50000
        else:
            self.max_lines_num = 20000

        self.multiplication_matrix = None
        self.addition_matrix = None
        self.mul1_full = None
        self.conditional_tiled = None
        self.mul_for_hypothetical = None
        self.add_for_hypothetical = None
        self.compute_repeated_computations()

        # For debugging purposes
        self.visualise_mask = visualise_mask
        self.mask_buffer_time_point = None

        # For obstruction mask (reset each time is called).
        self.empty_mask = self.chosen_math_library.ones((1500, 1500, 1), dtype=np.float64)

    def get_background_grating(self, frequency, linear=False):
        if linear:
            return self.linear_texture(frequency)
        else:
            return self.marble_texture()

    def linear_texture(self, frequency):
        """Simple linear repeating grating"""
        base_unit = self.chosen_math_library.concatenate((self.chosen_math_library.ones((1, frequency)),
                                                          self.chosen_math_library.zeros((1, frequency))), axis=1)
        number_required = int(self.width / frequency)
        full_width = self.chosen_math_library.tile(base_unit, number_required)[:, :self.width]
        full_arena = self.chosen_math_library.repeat(full_width, self.height, axis=0)
        full_arena = self.chosen_math_library.expand_dims(full_arena, 2)
        return full_arena

    def marble_texture(self):
        # TODO: Can be made much more efficient through none repeating computations.
        # Generate these randomly so grid can have any orientation.
        xPeriod = self.chosen_math_library.random.uniform(0.0, 10.0)
        yPeriod = self.chosen_math_library.random.uniform(0.0, 10.0)

        turbPower = 1.0
        turbSize = 162.0

        noise = self.chosen_math_library.absolute(self.chosen_math_library.random.randn(self.width, self.height))

        # TODO: Stop repeating the following:
        xp, yp = self.chosen_math_library.arange(self.width), self.chosen_math_library.arange(self.height)
        xy, py = self.chosen_math_library.meshgrid(xp, yp)
        xy = self.chosen_math_library.expand_dims(xy, 2)
        py = self.chosen_math_library.expand_dims(py, 2)
        coords = self.chosen_math_library.concatenate((xy, py), axis=2)

        xy_values = (coords[:, :, 0] * xPeriod / self.width) + (coords[:, :, 1] * yPeriod / self.height)
        size = turbSize

        # TODO: Stop repeating the following:
        turbulence = self.chosen_math_library.zeros((self.width, self.height))

        # TODO: Stop repeating the following:
        while size >= 1:
            reduced_coords = coords / size

            fractX = reduced_coords[:, :, 0] - reduced_coords[:, :, 0].astype(int)
            fractY = reduced_coords[:, :, 1] - reduced_coords[:, :, 1].astype(int)

            x1 = (reduced_coords[:, :, 0].astype(int) + self.width) % self.width
            y1 = (reduced_coords[:, :, 1].astype(int) + self.height) % self.height

            x2 = (x1 + self.width - 1) % self.width
            y2 = (y1 + self.height - 1) % self.height

            value = self.chosen_math_library.zeros((self.width, self.height))
            value += fractX * fractY * noise[y1, x1]
            value += (1 - fractX) * fractY * noise[y1, x2]
            value += fractX * (1 - fractY) * noise[y2, x1]
            value += (1 - fractX) * (1 - fractY) * noise[y2, x2]

            turbulence += value * size
            size /= 2.0

        turbulence = 128 * turbulence / turbSize
        xy_values += turbPower * turbulence / 256.0
        new_grating = 256 * self.chosen_math_library.abs(self.chosen_math_library.sin(xy_values * 3.14159))
        new_grating /= self.chosen_math_library.max(new_grating)  # Normalise
        new_grating = self.chosen_math_library.expand_dims(new_grating, 2)

        return new_grating

    def compute_repeated_computations(self):
        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        self.multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit, (self.max_lines_num, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.height - 1, self.width - 1])
        self.addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (self.max_lines_num, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        self.mul1_full = self.chosen_math_library.tile(mul1, (self.max_lines_num, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        self.conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (self.max_lines_num, 1))

        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (self.max_lines_num, 1, 1))

        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        self.add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (self.max_lines_num, 1, 1))

    def compute_repeated_computations_extended(self, num_lines):
        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit, (num_lines, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.height - 1, self.width - 1])
        addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (num_lines, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        mul1_full = self.chosen_math_library.tile(mul1, (num_lines, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (num_lines, 1))

        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (num_lines, 1, 1))

        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (num_lines, 1, 1))

        return multiplication_matrix, addition_matrix, mul1_full, mul_for_hypothetical, add_for_hypothetical, conditional_tiled

    def scatter(self, i, j, x, y):
        """Computes effects of absorption and scatter, but incorporates effect of implicit scatter from line spread."""
        positional_mask = (((x - i) ** 2 + (y - j) ** 2) ** 0.5)  # Measure of distance from fish at every point.
        desired_scatter = self.chosen_math_library.exp(-self.decay_rate * positional_mask)

        # To offset the effect of reduced sampling further away from fish (an unwanted effect that yielded massive
        # performance improvements). Should counter them exactly.
        implicit_scatter = self.chosen_math_library.sin(self.photoreceptor_rf_size) * positional_mask
        implicit_scatter[implicit_scatter < 1] = 1

        adjusted_scatter = desired_scatter * implicit_scatter
        adjusted_scatter = self.chosen_math_library.expand_dims(adjusted_scatter, 2)
        return adjusted_scatter

    def create_obstruction_mask_lines_cupy(self, fish_position, prey_locations, predator_locations):
        """Must use both numpy and cupy as otherwise GPU runs out of memory."""
        # Reset empty mask
        self.empty_mask[:] = 1.0

        # Compute prey positions relative to fish (Prey_num, 2)
        prey_relative_positions = prey_locations - fish_position

        # Compute distances of prey from fish.(Prey_num)
        prey_distances = (prey_relative_positions[:, 0] ** 2 + prey_relative_positions[:, 1] ** 2) ** 0.5

        # Compute angular size of prey from fish position. (Prey_num)
        prey_half_angular_size = self.chosen_math_library.arctan(self.prey_radius / prey_distances)

        # Compute angle between fish and prey where x-axis=0, and positive values are in upper field. (Prey_num)
        prey_angles = self.chosen_math_library.arctan(prey_relative_positions[:, 1] / prey_relative_positions[:, 0])
        prey_angles = self.chosen_math_library.expand_dims(prey_angles, 1)
        prey_angles = self.chosen_math_library.repeat(prey_angles, 2, 1)

        # From prey angular sizes, compute angles of edges of prey. (Prey_num, 2)
        prey_rf_offsets = self.chosen_math_library.expand_dims(prey_half_angular_size, 1)
        prey_rf_offsets = self.chosen_math_library.repeat(prey_rf_offsets, 2, 1)
        prey_rf_offsets = prey_rf_offsets * self.chosen_math_library.array([-1, 1])
        prey_extremities = prey_angles + prey_rf_offsets

        # Number of lines to project through prey or predators, determined by width, height, and size of features. (1)
        n_lines_prey = self.compute_n(self.chosen_math_library.max(prey_half_angular_size) * 2, len(prey_locations), p=prey_half_angular_size)

        # Create array of angles between prey extremities to form lines. (Prey_num * n_lines_prey)
        interpolated_line_angles = self.chosen_math_library.linspace(prey_extremities[:, 0], prey_extremities[:, 1],
                                                                     n_lines_prey).flatten()

        # Computing how far along each line prey are.  (Prey_num * n_lines_prey)
        prey_distance_along = prey_distances + self.prey_radius
        prey_distance_along = self.chosen_math_library.expand_dims(prey_distance_along, 1)
        prey_distance_along = self.chosen_math_library.repeat(prey_distance_along, n_lines_prey, 1)
        prey_distance_along = self.chosen_math_library.swapaxes(prey_distance_along, 0, 1).flatten()

        # Repeat prey locations so matches dimensions of other variables  (Prey_num * n_lines_prey, 2)
        expanded_prey_locations = self.chosen_math_library.tile(prey_locations, (n_lines_prey, 1))

        # Find which prey are in left half of visual field (as angle convention requires adjustment)
        # (Prey_num * n_lines_prey)
        prey_on_left = (expanded_prey_locations[:, 0] < fish_position[0]) * self.chosen_math_library.pi

        # Assign to array to include predators if present
        distance_along = prey_distance_along
        features_on_left = prey_on_left

        if predator_locations.size != 0:
            # Do the same for predator features.
            predator_relative_positions = predator_locations - fish_position
            predator_distances = (predator_relative_positions[:, 0] ** 2 + predator_relative_positions[:, 1] ** 2) \
                                 ** 0.5
            predator_half_angular_size = self.chosen_math_library.arctan(self.predator_radius / predator_distances)

            predator_angles = self.chosen_math_library.arctan(
                predator_relative_positions[:, 1] / predator_relative_positions[:, 0])
            predator_angles_expanded = self.chosen_math_library.expand_dims(predator_angles, 1)
            predator_angles_expanded = self.chosen_math_library.repeat(predator_angles_expanded, 2, 1)

            predator_rf_offsets = self.chosen_math_library.expand_dims(predator_half_angular_size, 1)
            predator_rf_offsets = self.chosen_math_library.repeat(predator_rf_offsets, 2, 1)
            predator_rf_offsets = predator_rf_offsets * self.chosen_math_library.array([-1, 1])
            predator_extremities = predator_angles_expanded + predator_rf_offsets

            # Number of lines to project through prey or predators, determined by width, height, and size of features.
            n_lines_predator = self.compute_n(self.chosen_math_library.max(predator_half_angular_size) * 2, 1)
            predator_interpolated_line_angles = self.chosen_math_library.linspace(predator_extremities[:, 0],
                                                                                  predator_extremities[:, 1],
                                                                                  n_lines_predator).flatten()

            # Combine prey and predator line angles.
            interpolated_line_angles = self.chosen_math_library.concatenate(
                (interpolated_line_angles, predator_interpolated_line_angles),
                axis=0)

            # Computing how far along each line predators are.
            predator_distance_along = (predator_distances ** 2 + self.predator_radius ** 2) ** 0.5
            predator_distance_along = self.chosen_math_library.expand_dims(predator_distance_along, 1)
            predator_distance_along = self.chosen_math_library.repeat(predator_distance_along, int(n_lines_predator), 1)
            predator_distance_along = self.chosen_math_library.swapaxes(predator_distance_along, 0, 1).flatten()
            predator_distance_along = predator_distance_along + self.predator_radius
            distance_along = self.chosen_math_library.concatenate((distance_along, predator_distance_along), axis=0)

            expanded_predator_locations = self.chosen_math_library.tile(predator_locations, (n_lines_predator, 1))
            predators_on_left = (expanded_predator_locations[:, 0] < fish_position[0]) * self.chosen_math_library.pi
            features_on_left = self.chosen_math_library.concatenate((features_on_left, predators_on_left), 0)

        total_lines = interpolated_line_angles.shape[0]

        # Ensure all angles are in designated range
        interpolated_line_angles_scaling = (interpolated_line_angles // (
                    self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        # Compute m using tan (N_obj x n)
        m = self.chosen_math_library.tan(interpolated_line_angles)

        # Compute c (N_obj*n)
        c = -m * fish_position[0]
        c = c + fish_position[1]

        # Compute components of intersections (N_obj*n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 1)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 1)

        # Slicing repeated matrices so have correct dimensions. (N_obj * n)
        if total_lines > self.max_lines_num:
            print("Too few lines... Attempting to create more.")

            multiplication_matrix = self.multiplication_matrix[:total_lines]
            addition_matrix = self.addition_matrix[:total_lines]
            mul1_full = self.mul1_full[:total_lines]
            mul_for_hypothetical = self.mul_for_hypothetical[:total_lines]
            add_for_hypothetical = self.add_for_hypothetical[:total_lines]
            conditional_tiled = self.conditional_tiled[:total_lines]
        else:
            multiplication_matrix, addition_matrix, mul1_full, mul_for_hypothetical, add_for_hypothetical, \
                conditional_tiled = self.compute_repeated_computations_extended(num_lines=total_lines)

        # Operations to compute all intersections of lines found
        m_mul = self.chosen_math_library.expand_dims(m, 1)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 1)
        m_mul = full_m * mul1_full

        m_mul[:, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, 1] = 1
        division_matrix[:, 3] = 1

        intersection_components = ((c_exp * multiplication_matrix) + addition_matrix) / division_matrix
        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 2)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 2)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (N_obj n 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]

        # Compute angles from the valid intersections (to find those matching original angles) (N_obj x 2)
        possible_vectors = valid_intersection_coordinates - fish_position
        computed_angles = self.chosen_math_library.arctan2(possible_vectors[:, 1], possible_vectors[:, 0])

        # Make sure angles are in correct range.
        angles_scaling = (computed_angles // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        computed_angles = computed_angles + angles_scaling

        computed_angles = self.chosen_math_library.round(computed_angles, 2)

        # Add adjustment for features appearing in left of visual field (needed because of angles convention)
        interpolated_line_angles = interpolated_line_angles + features_on_left

        # Make sure angles in correct range.
        interpolated_line_angles_scaling = (interpolated_line_angles // (
                    self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        # Get original angles in correct format.
        original_angles = self.chosen_math_library.round(interpolated_line_angles, 2)
        original_angles = self.chosen_math_library.expand_dims(original_angles, 1)
        original_angles = self.chosen_math_library.repeat(original_angles, 2, 1).flatten()

        # Check which angles match (should be precisely half)
        same_values = (computed_angles == original_angles) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]

        # Finding coordinates of object extremities.
        proj_vector = selected_intersections - fish_position
        proj_distance = (proj_vector[:, 0] ** 2 + proj_vector[:, 1] ** 2) ** 0.5

        # Compute fraction along the projection vector at which features lie
        try:
            fraction_along = distance_along / proj_distance
        except ValueError:
            # Note, think this error happened due to n being halved when more lines were required than were possible...
            print(f"Distance along dimensions: {distance_along.shape}")
            print(f"Proj distance dimensions: {proj_distance.shape}")
            print(f"Fish position: {fish_position}")
            print(f"Prey positions: {prey_locations}")
            print(f"Predator position: {predator_locations}")
            fraction_along = distance_along / proj_distance[:distance_along.shape[0], :]

        fraction_along = self.chosen_math_library.expand_dims(fraction_along, 1)
        fraction_along = self.chosen_math_library.repeat(fraction_along, 2, 1)

        # Find coordinates of lines on features.
        points_on_features = proj_vector * fraction_along
        points_on_features = fish_position + points_on_features
        points_on_features = self.chosen_math_library.expand_dims(points_on_features, 1)

        selected_intersections = self.chosen_math_library.reshape(selected_intersections, (total_lines, 1, 2))

        # Combine the valid wall intersection coordinates with the points found on features (total_lines, 2, 2)
        vertices = self.chosen_math_library.concatenate((selected_intersections, points_on_features), 1)
        vertices_xvals = vertices[:, :, 0]
        vertices_yvals = vertices[:, :, 1]

        # INTERPOLATION  - Get max and min vals in all vertices
        min_x = self.chosen_math_library.min(vertices_xvals, axis=1)
        max_x = self.chosen_math_library.max(vertices_xvals, axis=1)
        min_y = self.chosen_math_library.min(vertices_yvals, axis=1)
        max_y = self.chosen_math_library.max(vertices_yvals, axis=1)

        # SEGMENT COMPUTATION  - Compute length of x and y ranges to test
        x_lens = self.chosen_math_library.max(max_x - min_x)
        y_lens = self.chosen_math_library.max(max_y - min_y)

        x_len = self.chosen_math_library.around(x_lens)
        y_len = self.chosen_math_library.around(y_lens)

        # Get arrays of x and y to interpolate for each line (all must have same length due to array size).
        x_ranges = self.chosen_math_library.linspace(min_x, max_x, int(x_len))
        y_ranges = self.chosen_math_library.linspace(min_y, max_y, int(y_len))

        # Interpolation of lines, both in x and y.
        y_values = (m * x_ranges) + c
        y_values = self.chosen_math_library.floor(y_values)
        set_1 = self.chosen_math_library.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = self.chosen_math_library.floor(x_values)
        set_2 = self.chosen_math_library.stack((x_values, y_ranges), axis=-1)
        full_set = self.chosen_math_library.vstack((set_1, set_2)).astype(int)

        full_set = full_set.reshape(-1, 2)

        try:
            self.empty_mask[full_set[:, 1], full_set[:, 0]] = 0.0
        except IndexError:
            full_set = self.chosen_math_library.clip(full_set, 0, self.width - 1)
            self.empty_mask[full_set[:, 1], full_set[:, 0]] = 0.0

        # For debugging:
        # try:
        #     mask[full_set[:, 1], full_set[:, 0]] = 0  # NOTE: Inverting x and y to match standard in program.
        # except IndexError:
        #     print("IndexError")
        #     full_set[full_set > 1499] = 1499
        #     full_set[full_set < 0] = 0
        #     mask[full_set[:, 1], full_set[:, 0]] = 0  # NOTE: Inverting x and y to match standard in program.
        #
        #     plt.imshow(mask)
        #     plt.scatter(prey_locations[:, 0], prey_locations[:, 1])
        #     plt.show()
        #     mask = None

        return self.empty_mask

    def create_obstruction_mask_lines_mixed(self, fish_position, prey_locations, predator_locations):
        """Must use both numpy and cupy as otherwise GPU runs out of memory."""
        # Reset empty mask
        self.empty_mask[:] = 1

        # Compute prey positions relative to fish (Prey_num, 2)
        prey_relative_positions = prey_locations - fish_position

        # Compute distances of prey from fish.(Prey_num)
        prey_distances = (prey_relative_positions[:, 0] ** 2 + prey_relative_positions[:, 1] ** 2) ** 0.5

        # Compute angular size of prey from fish position. (Prey_num)
        prey_half_angular_size = np.arctan(self.prey_radius / prey_distances)

        # Compute angle between fish and prey where x-axis=0, and positive values are in upper field. (Prey_num)
        prey_angles = np.arctan(prey_relative_positions[:, 1] / prey_relative_positions[:, 0])
        prey_angles = np.expand_dims(prey_angles, 1)
        prey_angles = np.repeat(prey_angles, 2, 1)

        # From prey angular sizes, compute angles of edges of prey. (Prey_num, 2)
        prey_rf_offsets = np.expand_dims(prey_half_angular_size, 1)
        prey_rf_offsets = np.repeat(prey_rf_offsets, 2, 1)
        prey_rf_offsets = prey_rf_offsets * np.array([-1, 1])
        prey_extremities = prey_angles + prey_rf_offsets

        # Number of lines to project through prey or predators, determined by width, height, and size of features. (1)
        n_lines_prey = self.compute_n(np.max(prey_half_angular_size) * 2, len(prey_locations))

        # Create array of angles between prey extremities to form lines. (Prey_num * n_lines_prey)
        interpolated_line_angles = np.linspace(prey_extremities[:, 0], prey_extremities[:, 1], n_lines_prey).flatten()

        # Computing how far along each line prey are.  (Prey_num * n_lines_prey)
        prey_distance_along = prey_distances + self.prey_radius
        prey_distance_along = np.expand_dims(prey_distance_along, 1)
        prey_distance_along = np.repeat(prey_distance_along, n_lines_prey, 1)
        prey_distance_along = np.swapaxes(prey_distance_along, 0, 1).flatten()

        # Repeat prey locations so matches dimensions of other variables  (Prey_num * n_lines_prey, 2)
        expanded_prey_locations = np.tile(prey_locations, (n_lines_prey, 1))

        # Find which prey are in left half of visual field (as angle convention requires adjustment)
        # (Prey_num * n_lines_prey)
        prey_on_left = (expanded_prey_locations[:, 0] < fish_position[0]) * np.pi

        # Assign to array to include predators if present
        distance_along = prey_distance_along
        features_on_left = prey_on_left

        if predator_locations.size != 0:
            # Do the same for predator features.
            predator_relative_positions = predator_locations - fish_position
            predator_distances = (predator_relative_positions[:, 0] ** 2 + predator_relative_positions[:, 1] ** 2) \
                                 ** 0.5
            predator_half_angular_size = np.arctan(self.predator_radius / predator_distances)

            predator_angles = np.arctan(predator_relative_positions[:, 1] / predator_relative_positions[:, 0])
            predator_angles_expanded = np.expand_dims(predator_angles, 1)
            predator_angles_expanded = np.repeat(predator_angles_expanded, 2, 1)

            predator_rf_offsets = np.expand_dims(predator_half_angular_size, 1)
            predator_rf_offsets = np.repeat(predator_rf_offsets, 2, 1)
            predator_rf_offsets = predator_rf_offsets * np.array([-1, 1])
            predator_extremities = predator_angles_expanded + predator_rf_offsets

            # Number of lines to project through prey or predators, determined by width, height, and size of features.
            n_lines_predator = self.compute_n(np.max(predator_half_angular_size) * 2, 1)
            predator_interpolated_line_angles = np.linspace(predator_extremities[:, 0], predator_extremities[:, 1],
                                                            n_lines_predator).flatten()

            # Combine prey and predator line angles.
            interpolated_line_angles = np.concatenate((interpolated_line_angles, predator_interpolated_line_angles),
                                                      axis=0)

            # Computing how far along each line predators are.
            predator_distance_along = (predator_distances ** 2 + self.predator_radius ** 2) ** 0.5
            predator_distance_along = np.expand_dims(predator_distance_along, 1)
            predator_distance_along = np.repeat(predator_distance_along, int(n_lines_predator), 1)
            predator_distance_along = np.swapaxes(predator_distance_along, 0, 1).flatten()
            predator_distance_along = predator_distance_along + self.predator_radius
            distance_along = np.concatenate((distance_along, predator_distance_along), axis=0)

            expanded_predator_locations = np.tile(predator_locations, (n_lines_predator, 1))
            predators_on_left = (expanded_predator_locations[:, 0] < fish_position[0]) * np.pi
            features_on_left = np.concatenate((features_on_left, predators_on_left), 0)

        total_lines = interpolated_line_angles.shape[0]

        # Ensure all angles are in designated range
        interpolated_line_angles_scaling = (interpolated_line_angles // (np.pi * 2)) * np.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        # Compute m using tan (N_obj x n)
        m = np.tan(interpolated_line_angles)

        # Compute c (N_obj*n)
        c = -m * fish_position[0]
        c = c + fish_position[1]

        # Compute components of intersections (N_obj*n x 4)
        c_exp = np.expand_dims(c, 1)
        c_exp = np.repeat(c_exp, 4, 1)

        # Slicing repeated matrices so have correct dimensions. (N_obj * n)
        multiplication_matrix = self.multiplication_matrix[:total_lines]
        addition_matrix = self.addition_matrix[:total_lines]
        mul1_full = self.mul1_full[:total_lines]
        mul_for_hypothetical = self.mul_for_hypothetical[:total_lines]
        add_for_hypothetical = self.add_for_hypothetical[:total_lines]

        # Operations to compute all intersections of lines found
        m_mul = np.expand_dims(m, 1)
        full_m = np.repeat(m_mul, 4, 1)
        m_mul = full_m * mul1_full
        m_mul[:, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, 1] = 1
        division_matrix[:, 3] = 1

        intersection_components = ((c_exp * multiplication_matrix) + addition_matrix) / division_matrix
        intersection_coordinates = np.expand_dims(intersection_components, 2)
        intersection_coordinates = np.repeat(intersection_coordinates, 2, 2)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (N_obj n 2 x 2 x 2)
        conditional_tiled = self.conditional_tiled[:total_lines]
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]

        # Compute angles from the valid intersections (to find those matching original angles) (N_obj x 2)
        possible_vectors = valid_intersection_coordinates - fish_position
        computed_angles = np.arctan2(possible_vectors[:, 1], possible_vectors[:, 0])

        # Make sure angles are in correct range.
        angles_scaling = (computed_angles // (np.pi * 2)) * np.pi * -2
        computed_angles = computed_angles + angles_scaling

        computed_angles = np.round(computed_angles, 2)

        # Add adjustment for features appearing in left of visual field (needed because of angles convention)
        interpolated_line_angles = interpolated_line_angles + features_on_left

        # Make sure angles in correct range.
        interpolated_line_angles_scaling = (interpolated_line_angles // (np.pi * 2)) * np.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        # Get original angles in correct format.
        original_angles = np.round(interpolated_line_angles, 2)
        original_angles = np.expand_dims(original_angles, 1)
        original_angles = np.repeat(original_angles, 2, 1).flatten()

        # Check which angles match (should be precisely half)
        same_values = (computed_angles == original_angles) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]

        # Finding coordinates of object extremities.
        proj_vector = selected_intersections - fish_position
        proj_distance = (proj_vector[:, 0] ** 2 + proj_vector[:, 1] ** 2) ** 0.5

        # Compute fraction along the projection vector at which features lie
        fraction_along = distance_along / proj_distance
        # try:
        #     fraction_along = distance_along/proj_distance
        # except ValueError:
        #     print(f"Distance along dimensions: {distance_along.shape}")
        #     print(f"Proj distance dimensions: {proj_distance.shape}")
        #     print(f"Fish position: {fish_position}")
        #     print(f"Prey positions: {prey_locations}")
        #     print(f"Predator position: {predator_locations}")

        fraction_along = np.expand_dims(fraction_along, 1)
        fraction_along = np.repeat(fraction_along, 2, 1)

        # Find coordinates of lines on features.
        points_on_features = proj_vector * fraction_along
        points_on_features = fish_position + points_on_features
        points_on_features = np.expand_dims(points_on_features, 1)

        selected_intersections = np.reshape(selected_intersections, (total_lines, 1, 2))

        # Combine the valid wall intersection coordinates with the points found on features (total_lines, 2, 2)
        vertices = np.concatenate((selected_intersections, points_on_features), 1)
        vertices_xvals = vertices[:, :, 0]
        vertices_yvals = vertices[:, :, 1]

        # INTERPOLATION  - Get max and min vals in all vertices
        min_x = np.min(vertices_xvals, axis=1)
        max_x = np.max(vertices_xvals, axis=1)
        min_y = np.min(vertices_yvals, axis=1)
        max_y = np.max(vertices_yvals, axis=1)

        # SEGMENT COMPUTATION  - Compute length of x and y ranges to test
        x_lens = np.max(max_x - min_x)
        y_lens = np.max(max_y - min_y)

        x_len = round(x_lens)
        y_len = round(y_lens)

        # Get arrays of x and y to interpolate for each line (all must have same length due to array size).
        x_ranges = np.linspace(min_x, max_x, x_len)
        y_ranges = np.linspace(min_y, max_y, y_len)

        # Interpolation of lines, both in x and y.
        y_values = (m * x_ranges) + c
        y_values = np.floor(y_values)
        set_1 = np.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = np.floor(x_values)
        set_2 = np.stack((x_values, y_ranges), axis=-1)
        full_set = np.vstack((set_1, set_2)).astype(int)
        full_set = self.chosen_math_library.array(full_set)

        full_set = full_set.reshape(-1, 2)

        self.empty_mask[full_set[:, 1], full_set[:, 0]] = 0

        # For debugging:
        # try:
        #     mask[full_set[:, 1], full_set[:, 0]] = 0  # NOTE: Inverting x and y to match standard in program.
        # except IndexError:
        #     print("IndexError")
        #     full_set[full_set > 1499] = 1499
        #     full_set[full_set < 0] = 0
        #     mask[full_set[:, 1], full_set[:, 0]] = 0  # NOTE: Inverting x and y to match standard in program.
        #
        #     plt.imshow(mask)
        #     plt.scatter(prey_locations[:, 0], prey_locations[:, 1])
        #     plt.show()
        #     mask = None

        return self.empty_mask

    def get_luminance_mask(self, dark_light_ratio, dark_gain):
        dark_field_length = int(self.width * dark_light_ratio)
        luminance_mask = self.chosen_math_library.ones((self.width, self.height, 1))
        if self.light_gradient > 0 and dark_field_length > 0:
            luminance_mask[:dark_field_length, :, :] *= dark_gain
            luminance_mask[dark_field_length:, :, :] *= self.light_gain
            gradient = self.chosen_math_library.linspace(dark_gain, self.light_gain, self.light_gradient)
            gradient = self.chosen_math_library.expand_dims(gradient, 1)
            gradient = self.chosen_math_library.repeat(gradient, self.height, 1)
            gradient = self.chosen_math_library.expand_dims(gradient, 2)
            luminance_mask[int(dark_field_length-(self.light_gradient/2)):int(dark_field_length+(self.light_gradient/2)), :, :] = gradient

        else:
            luminance_mask[:dark_field_length, :, :] *= dark_gain
            luminance_mask[dark_field_length:, :, :] *= self.light_gain

        return luminance_mask

    def get_masked_pixels(self, fish_position, prey_locations, predator_locations, return_masks=False):
        """
        Returns masked pixels in form W.H.3
        With Red.UV.Red2
        """

        # Arena with features
        # A = self.chosen_math_library.array(np.concatenate((self.db[:, :, :1], self.db[:, :, 2:]), axis=2))
        A = self.chosen_math_library.concatenate((self.db[:, :, :1], self.db[:, :, 2:]), axis=2)

        # Combine with background grating
        AB = self.chosen_math_library.concatenate((A, self.background_grating), axis=2)

        # Get the luminance mask
        L = self.luminance_mask

        if prey_locations.size + predator_locations.size == 0:
            O = self.chosen_math_library.ones((self.width, self.height, 1), dtype=np.float64)
        else:
            O = self.create_obstruction_mask_lines_cupy(self.chosen_math_library.array(fish_position),
                                                        self.chosen_math_library.array(prey_locations),
                                                        self.chosen_math_library.array(predator_locations))
            # O = self.create_obstruction_mask_lines_mixed(fish_position, prey_locations, predator_locations)

        if self.differential_occlusion_gain:
            occluded = O[:, :, 0] == 0.0
            O = self.chosen_math_library.concatenate((O, O, O), axis=2)
            O[:, :, 0] += occluded * self.red_occlusion_gain
            O[:, :, 1] += occluded * self.uv_occlusion_gain
            O[:, :, 2] += occluded * self.red2_occlusion_gain
        else:
            O[O == 0] = self.red_occlusion_gain

        S = self.scatter(self.xp[:, None], self.yp[None, :], fish_position[1], fish_position[0])

        # AV = self.chosen_math_library.concatenate(
        #     (AB[:, :, 0:1], np.array(self.db[:, :, 1:2]), AB[:, :, 1:2]),
        #     axis=2)
        # G = AV * L * O * S
        # plt.imshow(self.background_grating)
        # plt.show()

        if self.visualise_mask:
            if self.visualise_mask == "O":
                self.mask_buffer_time_point = O
            elif self.visualise_mask == "G":
                AV = self.chosen_math_library.concatenate(
                    (AB[:, :, 0:1], self.chosen_math_library.array(self.db[:, :, 1:2]), AB[:, :, 1:2]), axis=2)
                G = AV * L * O * S
                self.mask_buffer_time_point = G
            else:
                print("Incorrect mask selected for saving")
            if self.using_gpu:
                self.mask_buffer_time_point = self.mask_buffer_time_point.get()
        # x = (AB * L * O * S)[:, :, 1]
        if return_masks:
            return AB, L, O, S
        else:
            return AB * L * O * S

    def compute_n(self, angular_size, number_of_this_feature, max_separation=1, p=None):
        max_dist = (self.width ** 2 + self.height ** 2) ** 0.5
        theta_separation = math.asin(max_separation / max_dist)
        n = (angular_size / theta_separation)

        if n * number_of_this_feature > self.max_lines_num:
            print(f"""Max lines num needs increase:
            Max lines num: {self.max_lines_num}
            Required lines for this feature alone: {n * number_of_this_feature}
            """)
            n = (self.max_lines_num * 0.8)/number_of_this_feature

        return int(n)

    def reset(self):
        """To be called at start of episode"""
        self.background_grating = self.get_background_grating(0)

    def erase(self, bkg=0):
        if bkg == 0:
            self.db = self.chosen_math_library.copy(self.base_db)
        else:
            self.db = self.chosen_math_library.copy(self.base_db_illuminated)

    def erase_visualisation(self, bkg=0.1):
        if bkg == 0:
            db = self.chosen_math_library.zeros((self.height, self.width, 3), dtype=np.double)
        else:
            db = (self.chosen_math_library.ones((self.height, self.width, 3), dtype=np.double) * bkg)
        db[1:2, :] = self.chosen_math_library.array([1, 0, 0])
        db[self.width - 2:self.width - 1, :] = self.chosen_math_library.array([1, 0, 0])
        db[:, 1:2] = self.chosen_math_library.array([1, 0, 0])
        db[:, self.height - 2:self.height - 1] = self.chosen_math_library.array([1, 0, 0])
        self.db_visualisation = db

    def get_base_arena(self, bkg=0.0):
        if bkg == 0:
            db = self.chosen_math_library.zeros((self.height, self.width, 3), dtype=np.double)
        else:
            db = (self.chosen_math_library.ones((self.height, self.width, 3), dtype=np.double) * bkg) / self.light_gain
        db[1:2, :] = self.chosen_math_library.array([1, 0, 0])
        db[self.width - 2:self.width - 1, :] = self.chosen_math_library.array([1, 0, 0])
        db[:, 1:2] = self.chosen_math_library.array([1, 0, 0])
        db[:, self.height - 2:self.height - 1] = self.chosen_math_library.array([1, 0, 0])
        return db

    def draw_walls(self):
        self.db[0:2, :] = [1, 0, 0]
        self.db[self.width - 1, :] = [1, 0, 0]
        self.db[:, 0] = [1, 0, 0]
        self.db[:, self.height - 1] = [1, 0, 0]

    def apply_light(self, dark_col, dark_gain, light_gain, visualisation):
        if dark_col < 0:
            dark_col = 0
        if visualisation:
            if self.light_gradient > 0 and dark_col > 0:
                gradient = self.chosen_math_library.linspace(dark_gain, light_gain, self.light_gradient)
                gradient = self.chosen_math_library.expand_dims(gradient, 0)
                gradient = self.chosen_math_library.repeat(gradient, self.height, 0)
                gradient = self.chosen_math_library.expand_dims(gradient, 2)
                self.db_visualisation[:, int(dark_col-(self.light_gradient/2)):int(dark_col+(self.light_gradient/2))] *= gradient
                self.db_visualisation[:, :int(dark_col-(self.light_gradient/2))] *= dark_gain
                self.db_visualisation[:, int(dark_col+(self.light_gradient/2)):] *= light_gain
            else:
                self.db_visualisation[:, :dark_col] *= dark_gain
                self.db_visualisation[:, dark_col:] *= light_gain

        else:
            if self.light_gradient > 0 and dark_col > 0:
                gradient = self.chosen_math_library.linspace(dark_gain, light_gain, self.light_gradient)
                gradient = self.chosen_math_library.expand_dims(gradient, 0)
                gradient = self.chosen_math_library.repeat(gradient, self.height, 0)
                gradient = self.chosen_math_library.expand_dims(gradient, 2)
                self.db[:, int(dark_col-(self.light_gradient/2)):int(dark_col+(self.light_gradient/2))] *= gradient
                self.db[:, :int(dark_col-(self.light_gradient/2))] *= dark_gain
                self.db[:, int(dark_col+(self.light_gradient/2)):] *= light_gain
            else:
                self.db[:, :dark_col] *= dark_gain
                self.db[:, dark_col:] *= light_gain

    def circle(self, center, rad, color, visualisation=False):
        rr, cc = draw.circle(center[1], center[0], rad, self.db.shape)
        if visualisation:
            self.db_visualisation[rr, cc, :] = color
        else:
            self.db[rr, cc, :] = color

    def show_salt_location(self, location):
        rr, cc = draw.circle(location[1], location[0], 10, self.db.shape)
        self.db_visualisation[rr, cc, :] = (0, 1, 0)

    def tail(self, head, left, right, tip, color, visualisation):
        tail_coordinates = np.array((head, left, tip, right))
        rr, cc = draw.polygon(tail_coordinates[:, 1], tail_coordinates[:, 0], self.db.shape)
        if visualisation:
            self.db_visualisation[rr, cc, :] = color
        else:
            self.db[rr, cc, :] = color

    def fish_shape(self, mouth_centre, mouth_rad, head_rad, tail_length, mouth_colour, body_colour, angle):
        offset = np.pi / 2
        angle += offset
        angle = -angle
        self.circle(mouth_centre, mouth_rad, mouth_colour, visualisation=True)  # For the mouth.
        dx1, dy1 = head_rad * np.sin(angle), head_rad * np.cos(angle)
        head_centre = (mouth_centre[0] + dx1,
                       mouth_centre[1] + dy1)
        self.circle(head_centre, head_rad, body_colour, visualisation=True)
        dx2, dy2 = -1 * dy1, dx1
        left_flank = (head_centre[0] + dx2,
                      head_centre[1] + dy2)
        right_flank = (head_centre[0] - dx2,
                       head_centre[1] - dy2)
        tip = (mouth_centre[0] + (tail_length + head_rad) * np.sin(angle),
               mouth_centre[1] + (tail_length + head_rad) * np.cos(angle))
        self.tail(head_centre, left_flank, right_flank, tip, body_colour, visualisation=True)

    def create_screen(self, fish_position, distance, colour):
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 10)
        self.db[rr, cc, :] = colour
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 9)
        self.db[rr, cc, :] = colour
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 8)
        self.db[rr, cc, :] = colour

    def vegetation(self, vertex, edge_size, color, visualisation=False):
        coordinates = np.array(((vertex[1], vertex[0]),
                                (vertex[1], vertex[0] + edge_size),
                                (vertex[1] + edge_size / 2, vertex[0] + edge_size - edge_size / 3),
                                (vertex[1] + edge_size, vertex[0] + edge_size),
                                (vertex[1] + edge_size, vertex[0]),
                                (vertex[1] + edge_size / 2, vertex[0] + edge_size / 3)))

        rr, cc = draw.polygon(coordinates[:, 0], coordinates[:, 1], self.db.shape)
        if visualisation:
            self.db_visualisation[rr, cc, :] = color
        else:
            self.db[rr, cc, :] = color

    @staticmethod
    def multi_circles(cx, cy, rad):
        rr, cc = draw.circle(0, 0, rad)
        rrs = np.tile(rr, (len(cy), 1)) + np.tile(np.reshape(cy, (len(cy), 1)), (1, len(rr)))
        ccs = np.tile(cc, (len(cx), 1)) + np.tile(np.reshape(cx, (len(cx), 1)), (1, len(cc)))
        return rrs, ccs

    def show_action_continuous(self, impulse, angle, fish_angle, x_position, y_position, colour):
        # rr, cc = draw.ellipse(int(y_position), int(x_position), (abs(angle) * 3) + 3, (impulse*0.5) + 3, rotation=-fish_angle)
        rr, cc = draw.ellipse(int(y_position), int(x_position), 3, (impulse * 0.5) + 3, rotation=-fish_angle)
        self.db_visualisation[rr, cc, :] = colour

    def show_action_discrete(self, fish_angle, x_position, y_position, colour):
        rr, cc = draw.ellipse(int(y_position), int(x_position), 5, 3, rotation=-fish_angle)
        self.db_visualisation[rr, cc, :] = colour

    def line(self, p1, p2, color):
        rr, cc = draw.line(p1[1], p1[0], p2[1], p2[0])
        self.db[rr, cc, :] = color

    def get_size(self):
        return self.width, self.height

    def show(self):
        if self.using_gpu:
            io.imshow(self.db.get())
        else:
            io.imshow(self.db)

        io.show()


if __name__ == "__main__":
    d = NewDrawingBoard(500, 500)
    d.circle((100, 200), 100, (1, 0, 0))
    d.line((50, 50), (100, 200), (0, 1, 0))
    d.show()
