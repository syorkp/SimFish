import numpy as np
import math
import matplotlib.pyplot as plt
from time import time

import skimage.draw as draw
from skimage import io


class FieldOfView:

    def __init__(self, local_dim, max_visual_distance, env_width, env_height):
        self.local_dim = local_dim
        self.max_visual_distance = max_visual_distance
        self.env_width = env_width
        self.env_height = env_height

        self.full_fov_top = None
        self.full_fov_bottom = None
        self.full_fov_left = None
        self.full_fov_right = None

        self.local_fov_top = None
        self.local_fov_bottom = None
        self.local_fov_left = None
        self.local_fov_right = None

        self.enclosed_fov_top = None
        self.enclosed_fov_bottom = None
        self.enclosed_fov_left = None
        self.enclosed_fov_right = None

    def update_field_of_view(self, fish_position):
        fish_position = np.round(fish_position).astype(int)

        self.full_fov_top = fish_position[1] - self.max_visual_distance
        self.full_fov_bottom = fish_position[1] + self.max_visual_distance + 1
        self.full_fov_left = fish_position[0] - self.max_visual_distance
        self.full_fov_right = fish_position[0] + self.max_visual_distance + 1

        self.local_fov_top = 0
        self.local_fov_bottom = self.local_dim
        self.local_fov_left = 0
        self.local_fov_right = self.local_dim

        self.enclosed_fov_top = self.full_fov_top
        self.enclosed_fov_bottom = self.full_fov_bottom
        self.enclosed_fov_left = self.full_fov_left
        self.enclosed_fov_right = self.full_fov_right

        if self.full_fov_top < 0:
            self.enclosed_fov_top = 0
            self.local_fov_top = -self.full_fov_top

        if self.full_fov_bottom > self.env_width:
            self.enclosed_fov_bottom = self.env_width
            self.local_fov_bottom = self.local_dim - (self.full_fov_bottom - self.env_width)

        if self.full_fov_left < 0:
            self.enclosed_fov_left = 0
            self.local_fov_left = -self.full_fov_left

        if self.full_fov_right > self.env_height:
            self.enclosed_fov_right = self.env_height
            self.local_fov_right = self.local_dim - (self.full_fov_right - self.env_height)


class DrawingBoard:

    def __init__(self, arena_width, arena_height, uv_decay_rate, red_decay_rate, photoreceptor_rf_size, using_gpu,
                 visualise_mask, prey_size=4,
                 predator_size=100, visible_scatter=0.3, background_grating_frequency=50, dark_light_ratio=0.0,
                 dark_gain=0.01, light_gain=1.0, occlusion_gain=(1.0, 1.0, 1.0),
                 light_gradient=0, max_visual_distance=1500, show_background=True):

        self.using_gpu = using_gpu

        if using_gpu:
            import cupy as cp
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

        self.width = arena_width
        self.height = arena_height
        self.uv_decay_rate = uv_decay_rate
        self.red_decay_rate = red_decay_rate
        self.light_gain = light_gain
        self.light_gradient = light_gradient
        self.photoreceptor_rf_size = photoreceptor_rf_size
        # self.db = None
        # self.db_visualisation = None
        max_visual_distance = np.round(max_visual_distance).astype(np.int32)
        self.local_dim = max_visual_distance * 2 + 1
        self.max_visual_distance = max_visual_distance
        self.base_db = self.get_base_arena()
        self.base_db_illuminated = self.get_base_arena(visible_scatter)
        self.erase(visible_scatter)
        #        self.local_db = self.chosen_math_library.zeros((self.local_dim, self.local_dim, 3))

        self.global_background_grating = self.get_background_grating(background_grating_frequency)
        self.global_luminance_mask = self.get_luminance_mask(dark_light_ratio, dark_gain)
        self.local_scatter = self.get_local_scatter()

        self.prey_size = prey_size * 2
        self.prey_radius = prey_size
        self.predator_size = predator_size * 2
        self.predator_radius = predator_size

        self.occlusion_gain = occlusion_gain

        # if self.red_occlusion_gain != self.uv_occlusion_gain or self.uv_occlusion_gain != self.red2_occlusion_gain:
        #     self.differential_occlusion_gain = True
        # else:
        #     self.differential_occlusion_gain = False

        # self.xp, self.yp = self.chosen_math_library.arange(self.width), self.chosen_math_library.arange(self.height)

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
        self.empty_mask = self.chosen_math_library.ones((self.local_dim, self.local_dim, 1), dtype=np.float64)

        self.show_background = show_background

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

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.local_dim - 1, self.local_dim - 1])
        self.addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (self.max_lines_num, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        self.mul1_full = self.chosen_math_library.tile(mul1, (self.max_lines_num, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.local_dim - 1, self.local_dim - 1, self.local_dim - 1, self.local_dim - 1])
        self.conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (self.max_lines_num, 1))

        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (self.max_lines_num, 1, 1))

        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.local_dim - 1], [self.local_dim - 1, 0]])
        self.add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (self.max_lines_num, 1, 1))

    def compute_repeated_computations_extended(self, num_lines):
        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit, (num_lines, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.local_dim - 1, self.local_dim - 1])
        addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (num_lines, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        mul1_full = self.chosen_math_library.tile(mul1, (num_lines, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.local_dim - 1, self.local_dim - 1, self.local_dim - 1, self.local_dim - 1])
        conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (num_lines, 1))

        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (num_lines, 1, 1))

        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.local_dim - 1], [self.local_dim - 1, 0]])
        add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (num_lines, 1, 1))

        return multiplication_matrix, addition_matrix, mul1_full, mul_for_hypothetical, add_for_hypothetical, conditional_tiled

    def get_local_scatter(self):
        """Computes effects of absorption and scatter, but incorporates effect of implicit scatter from line spread."""

        x, y = self.chosen_math_library.arange(self.local_dim), self.chosen_math_library.arange(self.local_dim)
        y = self.chosen_math_library.expand_dims(y, 1)
        j = self.max_visual_distance + 1
        positional_mask = (((x - j) ** 2 + (y - j) ** 2) ** 0.5)  # Measure of distance from centre to every pixel
        desired_uv_scatter = self.chosen_math_library.exp(-self.uv_decay_rate * positional_mask)
        desired_red_scatter = self.chosen_math_library.exp(-self.red_decay_rate * positional_mask)
        # To offset the effect of reduced sampling further away from fish (an unwanted effect that yielded massive
        # performance improvements). Should counter them exactly.
        implicit_scatter = self.chosen_math_library.sin(self.photoreceptor_rf_size) * positional_mask
        implicit_scatter[implicit_scatter < 1] = 1

        adjusted_uv_scatter = desired_uv_scatter * implicit_scatter
        adjusted_red_scatter = desired_red_scatter * implicit_scatter
        adjusted_scatter = self.chosen_math_library.stack(
            [adjusted_red_scatter, adjusted_uv_scatter, adjusted_red_scatter],
            axis=2)
        # adjusted_scatter = self.chosen_math_library.expand_dims(adjusted_scatter, 2)
        return adjusted_scatter

    def create_obstruction_mask_lines_cupy(self, fish_position, prey_locations, predator_locations,
                                           prey_occlusion=False):
        """Must use both numpy and cupy as otherwise GPU runs out of memory. positions are board-centric"""
        # Reset empty mask
        self.empty_mask[:] = 1.0
        # fish position within FOV (central)
        fish_position_FOV = self.chosen_math_library.array([self.max_visual_distance, self.max_visual_distance])

        if predator_locations.size == 0 and (not prey_occlusion or prey_locations.size == 0):
            return self.empty_mask

        if prey_occlusion:
            # Compute prey positions already relative to fish (Prey_num, 2)
            prey_relative_positions = prey_locations - fish_position
            # Compute distances of prey from fish.(Prey_num)
            prey_distances = (prey_relative_positions[:, 0] ** 2 + prey_relative_positions[:, 1] ** 2) ** 0.5

            prey_relative_positions = prey_relative_positions[prey_distances < self.max_visual_distance, :]
            prey_distances = prey_distances[prey_distances < self.max_visual_distance]
            prey_locations_FOV = prey_relative_positions + self.max_visual_distance

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
            n_lines_prey = self.compute_n(self.chosen_math_library.max(prey_half_angular_size) * 2,
                                          len(prey_locations_FOV), p=prey_half_angular_size)

            # Create array of angles between prey extremities to form lines. (Prey_num * n_lines_prey)
            interpolated_line_angles = self.chosen_math_library.linspace(prey_extremities[:, 0], prey_extremities[:, 1],
                                                                         n_lines_prey).flatten()

            # Computing how far along each line prey are.  (Prey_num * n_lines_prey)
            prey_distance_along = prey_distances + self.prey_radius
            prey_distance_along = self.chosen_math_library.expand_dims(prey_distance_along, 1)
            prey_distance_along = self.chosen_math_library.repeat(prey_distance_along, n_lines_prey, 1)
            prey_distance_along = self.chosen_math_library.swapaxes(prey_distance_along, 0, 1).flatten()

            # Repeat prey locations so matches dimensions of other variables  (Prey_num * n_lines_prey, 2)
            expanded_prey_locations = self.chosen_math_library.tile(prey_locations_FOV, (n_lines_prey, 1))

            # Find which prey are in left half of visual field (as angle convention requires adjustment)
            # (Prey_num * n_lines_prey)
            prey_on_left = (expanded_prey_locations[:, 0] < fish_position_FOV[0]) * self.chosen_math_library.pi

            # Assign to array to include predators if present
            distance_along = prey_distance_along
            features_on_left = prey_on_left

        if predator_locations.size != 0:
            # Do the same for predator features.
            predator_relative_positions = predator_locations - fish_position
            predator_locations_FOV = predator_relative_positions + self.max_visual_distance
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
            if prey_occlusion:
                interpolated_line_angles = self.chosen_math_library.concatenate(
                    (interpolated_line_angles, predator_interpolated_line_angles),
                    axis=0)
            else:
                interpolated_line_angles = predator_interpolated_line_angles

            # Computing how far along each line predators are.
            predator_distance_along = (predator_distances ** 2 + self.predator_radius ** 2) ** 0.5
            predator_distance_along = self.chosen_math_library.expand_dims(predator_distance_along, 1)
            predator_distance_along = self.chosen_math_library.repeat(predator_distance_along, int(n_lines_predator), 1)
            predator_distance_along = self.chosen_math_library.swapaxes(predator_distance_along, 0, 1).flatten()
            predator_distance_along = predator_distance_along + self.predator_radius
            if prey_occlusion:
                distance_along = self.chosen_math_library.concatenate((distance_along, predator_distance_along), axis=0)
            else:
                distance_along = predator_distance_along

            expanded_predator_locations = self.chosen_math_library.tile(predator_locations_FOV, (n_lines_predator, 1))
            predators_on_left = (expanded_predator_locations[:, 0] < fish_position_FOV[0]) * self.chosen_math_library.pi

            if prey_occlusion:
                features_on_left = self.chosen_math_library.concatenate((features_on_left, predators_on_left), 0)
            else:
                features_on_left = predators_on_left

        # If no predators, and no prey occlusion, return now with empty obstruction mask
        if "interpolated_line_angles" not in locals():
            return self.empty_mask

        total_lines = interpolated_line_angles.shape[0]

        # Ensure all angles are in designated range
        interpolated_line_angles_scaling = (interpolated_line_angles // (
                self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        # Compute m using tan (N_obj x n)
        m = self.chosen_math_library.tan(interpolated_line_angles)

        # Compute c (N_obj*n)
        c = -m * fish_position_FOV[0]
        c = c + fish_position_FOV[1]

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
        possible_vectors = valid_intersection_coordinates - fish_position_FOV
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
        proj_vector = selected_intersections - fish_position_FOV
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
        points_on_features = fish_position_FOV + points_on_features
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
        try:
            x_lens = self.chosen_math_library.max(max_x - min_x)
        except ValueError:
            # In the event that all prey are further away than max visual distance.
            return self.empty_mask

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
            full_set = self.chosen_math_library.clip(full_set, 0, self.local_dim - 1)
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

        occluded = (self.empty_mask[:, :, 0] == 0.0)
        O = self.chosen_math_library.repeat(self.empty_mask, 3, 2)
        # O = self.chosen_math_library.concatenate((self.empty_mask, self.empty_mask, self.empty_mask), axis=2)
        O[:, :, 0] += occluded * self.occlusion_gain[0]
        O[:, :, 1] += occluded * self.occlusion_gain[1]
        O[:, :, 2] += occluded * self.occlusion_gain[2]

        return O

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

    def extend_A(self, A, FOV):
        """Extends the arena pixels (red1 channel), by stretching the values computed at the wall points along the
        whole FOV in that direction"""

        if FOV["full_fov"][0] < 0:
            low_dim_top = abs(FOV["full_fov"][0])
        else:
            low_dim_top = 0
        if FOV["full_fov"][2] < 0:
            low_dim_left = abs(FOV["full_fov"][2])
        else:
            low_dim_left = 0

        if FOV["full_fov"][1] > self.height:
            high_dim_bottom = abs(FOV["full_fov"][1]) - self.local_dim
        else:
            high_dim_bottom = self.local_dim
        if FOV["full_fov"][3] > self.width:
            high_dim_right = abs(FOV["full_fov"][3]) - self.local_dim
        else:
            high_dim_right = self.local_dim

        print(high_dim_bottom)
        print(high_dim_right)

        pixel_to_extend = A[low_dim_top, low_dim_left, 0]
        A[:, :low_dim_left, 0] = pixel_to_extend
        A[:low_dim_top, :, 0] = pixel_to_extend

        pixel_to_extend = A[high_dim_bottom, high_dim_right, 0]
        A[:, high_dim_right:, 0] = pixel_to_extend
        A[high_dim_bottom:, :, 0] = pixel_to_extend

        return A

    def get_masked_pixels(self, fish_position, prey_locations, predator_locations):
        """
        Returns masked pixels in form W.H.3
        With Red.UV.Red2
        """

        FOV = self.get_field_of_view(fish_position)

        A = self.chosen_math_library.array(self.local_db)

        # apply FOV portion of luminance mask
        local_luminance_mask = self.global_luminance_mask[FOV['enclosed_fov'][0]:FOV['enclosed_fov'][1],
                                                          FOV['enclosed_fov'][2]:FOV['enclosed_fov'][3], :]

        A[FOV['local_coordinates_fov'][0]:FOV['local_coordinates_fov'][1],
          FOV['local_coordinates_fov'][2]:FOV['local_coordinates_fov'][3], :] *= local_luminance_mask

        # If FOV extends outside the arena, extend the A image
        A = self.extend_A(A, FOV)

        if prey_locations.size + predator_locations.size == 0:
            O = self.chosen_math_library.ones((self.local_dim, self.local_dim, 3), dtype=np.float64)
        else:
            O = self.create_obstruction_mask_lines_cupy(self.chosen_math_library.array(fish_position),
                                                        self.chosen_math_library.array(prey_locations),
                                                        self.chosen_math_library.array(predator_locations))

        return A * O * self.local_scatter

    def compute_n(self, angular_size, number_of_this_feature, max_separation=1, p=None):

        theta_separation = math.asin(max_separation / self.max_visual_distance)
        n = (angular_size / theta_separation)

        if n * number_of_this_feature > self.max_lines_num:
            print(f"""Max lines num needs increase:
            Max lines num: {self.max_lines_num}
            Required lines for this feature alone: {n * number_of_this_feature}
            """)
            n = (self.max_lines_num * 0.8) / number_of_this_feature

        return int(n)

    def reset(self):
        """To be called at start of episode"""
        self.global_background_grating = self.get_background_grating(0)

        # Dont need to call each ep?
        # self.local_scatter = self.get_local_scatter()

    def erase(self, bkg=0):
        # self.local_db = self.chosen_math_library.zeros((self.local_dim, self.local_dim, 3))

        if bkg == 0:
            self.local_db = self.chosen_math_library.copy(self.base_db)
        else:
            self.local_db = self.chosen_math_library.copy(self.base_db_illuminated)

    def get_base_arena(self, bkg=0.0):
        if bkg == 0:
            db = self.chosen_math_library.zeros((self.local_dim, self.local_dim, 3), dtype=np.double)
        else:
            db = (self.chosen_math_library.ones((self.local_dim, self.local_dim, 3),
                                                dtype=np.double) * bkg) / self.light_gain
        return db

    def circle(self, center, rad, color, visualisation=False):
        rr, cc = draw.circle(center[1], center[0], rad, self.db.shape)
        if visualisation:
            self.db_visualisation[rr, cc, :] = color
        else:
            self.db[rr, cc, :] = color

    def show_salt_location(self, location):
        rr, cc = draw.circle(location[1], location[0], 10, self.db.shape)
        self.db_visualisation[rr, cc, :] = (1, 0, 0)

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

    def _draw_past_actions(self, n_actions_to_show):
        # Select subset of actions to show
        if len(self.action_buffer) > n_actions_to_show:
            actions_to_show = self.action_buffer[len(self.action_buffer) - n_actions_to_show:]
            positions_to_show = self.position_buffer[len(self.position_buffer) - n_actions_to_show:]
            fish_angles_to_show = self.fish_angle_buffer[len(self.fish_angle_buffer) - n_actions_to_show:]
        else:
            actions_to_show = self.action_buffer
            positions_to_show = self.position_buffer
            fish_angles_to_show = self.fish_angle_buffer

        for i, a in enumerate(actions_to_show):
            adjusted_colour_index = ((1 - self.env_variables["bkg_scatter"]) * (i + 1) / len(actions_to_show)) + \
                                    self.env_variables["bkg_scatter"]
            if self.continuous_actions:
                # action_colour = (1 * ((i+1)/len(actions_to_show)), 0, 0)
                if a[1] < 0:
                    action_colour = (
                    adjusted_colour_index, self.env_variables["bkg_scatter"], self.env_variables["bkg_scatter"])
                else:
                    action_colour = (self.env_variables["bkg_scatter"], adjusted_colour_index, adjusted_colour_index)

                self.board.show_action_continuous(a[0], a[1], fish_angles_to_show[i], positions_to_show[i][0],
                                                  positions_to_show[i][1], action_colour)
            else:
                action_colour = self.fish.get_action_colour(actions_to_show[i], adjusted_colour_index,
                                                            self.env_variables["bkg_scatter"])
                self.board.show_action_discrete(fish_angles_to_show[i], positions_to_show[i][0],
                                                positions_to_show[i][1], action_colour)

    def draw_walls(self, FOV):
        """Draws walls as deep into FOV beyond wall objects as possible."""

        self.local_db[FOV['local_coordinates_fov'][0], FOV['local_coordinates_fov'][2]:FOV['local_coordinates_fov'][3],
        0] = 1
        self.local_db[FOV['local_coordinates_fov'][1] - 1,
        FOV['local_coordinates_fov'][2]:FOV['local_coordinates_fov'][3], 0] = 1
        self.local_db[FOV['local_coordinates_fov'][0]:FOV['local_coordinates_fov'][1], FOV['local_coordinates_fov'][2],
        0] = 1
        self.local_db[FOV['local_coordinates_fov'][0]:FOV['local_coordinates_fov'][1],
        FOV['local_coordinates_fov'][3] - 1, 0] = 1

    def draw_shapes_environmental(self, visualisation, prey_pos, sand_grain_pos=np.array([]),
                                  sand_grain_colour=(0, 0, 1)):  # prey/sand positions are fish-centric
        # if visualisation:  # Only draw fish if in visualisation mode
        #     if self.env_variables["show_fish_body_energy_state"]:
        #         fish_body_colour = (1 - self.fish.energy_level, self.fish.energy_level, 0)
        #     else:
        #         fish_body_colour = self.fish.head.color

        #     self.fish_shape(self.fish.body.position, self.env_variables['fish_mouth_size'],
        #                           self.env_variables['fish_head_size'], self.env_variables['fish_tail_length'],
        #                           self.fish.mouth.color, fish_body_colour, self.fish.body.angle)

        prey_pos += self.max_visual_distance + 1  # fish-centric to fov-centric
        sand_grain_pos += self.max_visual_distance + 1

        # remove out of bounds prey
        prey_pos = prey_pos[np.all((np.all(prey_pos <= self.local_dim, axis=1),
                                    np.all(prey_pos >= 0, axis=1)), axis=0)]
        # remove out of bounds sand grains
        sand_grain_pos = sand_grain_pos[np.all((np.all(sand_grain_pos <= self.local_dim, axis=1),
                                                np.all(sand_grain_pos >= 0, axis=1)), axis=0)]

        if len(prey_pos) > 0:
            # px = np.round(np.array([pr.position[0] for pr in self.prey_bodies])).astype(int)
            # py = np.round(np.array([pr.position[1] for pr in self.prey_bodies])).astype(int)
            rrs, ccs = self.multi_circles(prey_pos[:, 0], prey_pos[:, 1], self.prey_size)
            rrs = np.clip(rrs, 0, self.local_dim - 1)
            ccs = np.clip(ccs, 0, self.local_dim - 1)

            #            try:
            #                if visualisation:
            #                    self.board.db_visualisation[rrs, ccs] = self.prey_shapes[0].color
            #                else:
            #                    self.board.db[rrs, ccs] = self.prey_shapes[0].color

            self.local_db[rrs, ccs, 1] = 1

            # except IndexError:
            #     print(f"Index Error for: PX: {max(rrs.flatten())}, PY: {max(ccs.flatten())}")
            #     if max(rrs.flatten()) > self.env_variables['height']:
            #         lost_index = np.argmax(py)
            #     elif max(ccs.flatten()) > self.env_variables['width']:
            #         lost_index = np.argmax(px)
            #     else:
            #         lost_index = 0
            #         print(f"Fix needs to be tuned: PX: {max(px)}, PY: {max(py)}")
            #     self.prey_bodies.pop(lost_index)
            #     self.prey_shapes.pop(lost_index)
            #     self.draw_shapes(visualisation=visualisation)

        if len(sand_grain_pos) > 0:
            rrs, ccs = self.multi_circles(sand_grain_pos[:, 0], sand_grain_pos[:, 1], self.prey_size)
            self.board.db[rrs, ccs] = sand_grain_colour

        # for i, pr in enumerate(self.predator_bodies):
        #     self.board.circle(pr.position, self.env_variables['predator_size'], self.predator_shapes[i].color, visualisation)

        # for i, pr in enumerate(self.vegetation_bodies):
        #     self.board.vegetation(pr.position, self.env_variables['vegetation_size'], self.vegetation_shapes[i].color, visualisation)

        # if self.predator_body is not None:
        #     if self.first_attack:
        #         self.board.circle(self.predator_body.position, self.loom_predator_current_size,
        #                           self.predator_shape.color, visualisation)
        #     else:
        #         self.board.circle(self.predator_body.position, self.env_variables['predator_size'],
        #                           self.predator_shape.color, visualisation)

        # # For displaying location of salt source
        # if visualisation:
        #     if self.env_variables["salt"] and self.env_variables["max_salt_damage"] > 0:
        #         self.board.show_salt_location(self.salt_location)

        # # For creating a screen around prey to test.
        # if self.background:
        #     if self.background == "Green":
        #         colour = (0, 1, 0)
        #     elif self.background == "Red":
        #         colour = (1, 0, 0)
        #     else:
        #         print("Invalid Background Colour")
        #         return
        #     self.board.create_screen(self.fish.body.position, self.env_variables["max_vis_dist"], colour)

    def draw_background(self, FOV):  # slice the global background for current field of view

        background_slice = self.global_background_grating[FOV['enclosed_fov'][0]:FOV['enclosed_fov'][1],
                           FOV['enclosed_fov'][2]:FOV['enclosed_fov'][3], 0]

        self.local_db[FOV['local_coordinates_fov'][0]:FOV['local_coordinates_fov'][1],
        FOV['local_coordinates_fov'][2]:FOV['local_coordinates_fov'][3], 2] = background_slice

    def get_field_of_view(self, fish_location):  # use field location to get field of view
        # top bottom left right
        fish_location = np.round(fish_location).astype(int)
        full_fov = [fish_location[1] - self.max_visual_distance,
                    fish_location[1] + self.max_visual_distance + 1,
                    fish_location[0] - self.max_visual_distance,
                    fish_location[0] + self.max_visual_distance + 1]

        # check if field of view is within bounds of global background
        local_coordinates_fov = [0, self.local_dim, 0, self.local_dim]
        enclosed_fov = full_fov.copy()
        if full_fov[0] < 0:
            enclosed_fov[0] = 0
            local_coordinates_fov[0] = -full_fov[0]
        if full_fov[1] > self.global_background_grating.shape[0]:
            enclosed_fov[1] = self.global_background_grating.shape[0]
            local_coordinates_fov[1] = self.local_dim - (full_fov[1] - self.global_background_grating.shape[0])
        if full_fov[2] < 0:
            enclosed_fov[2] = 0
            local_coordinates_fov[2] = -full_fov[2]
        if full_fov[3] > self.global_background_grating.shape[1]:
            enclosed_fov[3] = self.global_background_grating.shape[1]
            local_coordinates_fov[3] = self.local_dim - (full_fov[3] - self.global_background_grating.shape[1])

        return {'full_fov': full_fov, 'enclosed_fov': enclosed_fov, 'local_coordinates_fov': local_coordinates_fov}


if __name__ == "__main__":
    d = DrawingBoard(500, 500)
    d.circle((100, 200), 100, (1, 0, 0))
    d.line((50, 50), (100, 200), (0, 1, 0))
    d.show()
