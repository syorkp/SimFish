import numpy as np
import cupy as cp
import math

import matplotlib.pyplot as plt


class Eye:

    def __init__(self, board, verg_angle, retinal_field, is_left, env_variables, dark_col, using_gpu):
        # Use CUPY if using GPU.
        self.using_gpu = using_gpu
        if using_gpu:
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

        self.board = board
        self.dark_gain = env_variables['dark_gain']
        self.light_gain = env_variables['light_gain']
        self.bkg_scatter = env_variables['bkg_scatter']
        self.dark_col = dark_col
        self.dist = None
        self.theta = None
        self.width, self.height = self.board.get_size()
        self.retinal_field_size = retinal_field

        if env_variables['shared_photoreceptor_channels']:
            self.shared_photoreceptor_channels = True

            self.photoreceptor_num = env_variables['uv_photoreceptor_num']
            self.photoreceptor_rf_size = env_variables['uv_photoreceptor_rf_size']
            self.photoreceptor_spacing = self.retinal_field_size / self.photoreceptor_num
            self.max_photoreceptor_num = self.photoreceptor_num

            if env_variables['incorporate_uv_strike_zone']:
                self.photoreceptor_angles = self.update_angles_strike_zone(verg_angle, retinal_field, is_left, self.photoreceptor_num)
            else:
                self.photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left, self.photoreceptor_num)
            self.readings = self.chosen_math_library.zeros((self.photoreceptor_num, 2), 'int')

            self.n = self.compute_n(self.photoreceptor_rf_size)

        else:
            self.shared_photoreceptor_channels = False

            self.uv_photoreceptor_num = env_variables['uv_photoreceptor_num']
            self.red_photoreceptor_num = env_variables['red_photoreceptor_num']

            self.uv_photoreceptor_rf_size = env_variables['uv_photoreceptor_rf_size']
            self.red_photoreceptor_rf_size = env_variables['red_photoreceptor_rf_size']

            self.uv_photoreceptor_spacing = self.retinal_field_size / self.uv_photoreceptor_num
            self.red_photoreceptor_spacing = self.retinal_field_size / self.red_photoreceptor_num

            self.max_photoreceptor_num = max([self.uv_photoreceptor_num, self.red_photoreceptor_num])
            self.min_photoreceptor_num = min([self.uv_photoreceptor_num, self.red_photoreceptor_num])

            if env_variables['incorporate_uv_strike_zone']:
                self.uv_photoreceptor_angles = self.update_angles_strike_zone(verg_angle, retinal_field, is_left, self.uv_photoreceptor_num)
            else:
                self.uv_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left, self.uv_photoreceptor_num)
            self.red_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left, self.red_photoreceptor_num)

            self.uv_readings = self.chosen_math_library.zeros((env_variables['uv_photoreceptor_num'], 1), 'int')
            self.red_readings = self.chosen_math_library.zeros((env_variables['red_photoreceptor_num'], 1), 'int')

            self.n = self.compute_n(max([self.uv_photoreceptor_rf_size, self.red_photoreceptor_rf_size]))

            self.indices_for_padding = self.chosen_math_library.around(
                self.chosen_math_library.linspace(0, self.min_photoreceptor_num - 1, self.max_photoreceptor_num)).astype(int)

        # Compute repeated measures:
        self.channel_angles_surrounding = None
        self.channel_angles_surrounding_2 = None
        self.mul_for_hypothetical = None
        self.add_for_hypothetical = None
        self.mul1_full = None
        self.addition_matrix = None
        self.conditional_tiled = None
        self.multiplication_matrix = None
        self.get_repeated_computations()

    def get_repeated_computations(self):
        if self.shared_photoreceptor_channels:
            channel_angles_surrounding = self.chosen_math_library.expand_dims(self.photoreceptor_angles, 1)
            channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, self.n, 1)
            rf_offsets = self.chosen_math_library.linspace(-self.photoreceptor_rf_size / 2, self.photoreceptor_rf_size / 2, num=self.n)
            self.channel_angles_surrounding = channel_angles_surrounding + rf_offsets
        else:
            channel_angles_surrounding = self.chosen_math_library.expand_dims(self.uv_photoreceptor_angles, 1)
            channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, self.n, 1)
            rf_offsets = self.chosen_math_library.linspace(-self.uv_photoreceptor_rf_size / 2,
                                                           self.uv_photoreceptor_rf_size / 2, num=self.n)
            self.channel_angles_surrounding = channel_angles_surrounding + rf_offsets

            channel_angles_surrounding_2 = self.chosen_math_library.expand_dims(self.red_photoreceptor_angles, 1)
            channel_angles_surrounding_2 = self.chosen_math_library.repeat(channel_angles_surrounding_2, self.n, 1)
            rf_offsets_2 = self.chosen_math_library.linspace(-self.red_photoreceptor_rf_size / 2,
                                                           self.red_photoreceptor_rf_size / 2, num=self.n)
            self.channel_angles_surrounding_2 = channel_angles_surrounding_2 + rf_offsets_2

        # Same for both, just requires different dimensions
        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (self.max_photoreceptor_num, self.n, 1, 1))
        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        self.add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (self.max_photoreceptor_num, self.n, 1, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        self.mul1_full = self.chosen_math_library.tile(mul1, (self.max_photoreceptor_num, self.n, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.height - 1, self.width - 1])
        self.addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (self.max_photoreceptor_num, self.n, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        self.conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (self.max_photoreceptor_num, self.n, 1))

        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        self.multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit, (self.max_photoreceptor_num, self.n, 1))

    def update_angles(self, verg_angle, retinal_field, is_left, photoreceptor_num):
        """Set the eyes visual angles."""
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        return self.chosen_math_library.linspace(min_angle, max_angle, photoreceptor_num)

    def sample_half_normal_distribution(self, min_angle, max_angle, photoreceptor_num):
        # Problem is that fish will find it difficult to learn with such large trial-to-trial variablility in input
        # distribution, though maybe is realistic. Alternatively, can set random seed for this part?
        sampled_values = np.random.randn(photoreceptor_num)
        sampled_values = [abs(i) for i in sampled_values]
        scaling_factor = max(sampled_values)
        sampled_values = sampled_values / scaling_factor

        difference = abs(max_angle - min_angle)
        sampled_values = sampled_values * difference
        sampled_values = sampled_values - max_angle

        sampled_values = sampled_values * -1

        return np.array(sorted(sampled_values))

    def create_half_normal_distribution(self, min_angle, max_angle, photoreceptor_num):
        # IDEA: Could use normal distribution equation to produce a set of differences between values (by dividing array
        # of 1s by density), then generating difference.
        # TODO: Handling of mu and sigma
        # TODO: selection of range based on which parts of normal distribtuion to approximate.
        mu = 0
        sigma = 1
        angle_difference = abs(max_angle - min_angle)

        angle_range = np.linspace(min_angle, max_angle, photoreceptor_num)
        frequencies = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(angle_range-mu)**2/(2*sigma**2))
        differences = 1/frequencies
        differences[0] = 0
        total_difference = np.sum(differences)
        differences = (differences*angle_difference)/total_difference
        cumulative_differences = np.cumsum(differences)
        photoreceptor_angles = min_angle + cumulative_differences
        return photoreceptor_angles

    def update_angles_strike_zone(self, verg_angle, retinal_field, is_left, photoreceptor_num):
        """Set the eyes visual angles, with the option of particular distributions."""

        # Half normal distribution
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2

        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2

        # sampled_values = self.sample_half_normal_distribution(min_angle, max_angle, photoreceptor_num)
        computed_values = self.create_half_normal_distribution(min_angle, max_angle, photoreceptor_num)

        # plt.scatter(computed_values, computed_values)
        # plt.show()

        return self.chosen_math_library.array(computed_values)

    def read(self, masked_arena_pixels, eye_x, eye_y, fish_angle):
        if self.shared_photoreceptor_channels:
            # Angles with respect to fish (doubled) (PR_N x n)
            channel_angles_surrounding = self.channel_angles_surrounding + fish_angle
            self.readings = self._read(masked_arena_pixels, eye_x, eye_y, channel_angles_surrounding, self.photoreceptor_num)
        else:
            # TODO: Could be sped up by running both in parallel and splitting the results!
            # Angles with respect to fish (doubled) (PR_N x n)
            channel_angles_surrounding = self.channel_angles_surrounding + fish_angle
            self.uv_readings = self._read(masked_arena_pixels[:, :, 1:], eye_x, eye_y, channel_angles_surrounding, self.uv_photoreceptor_num)

            # Angles with respect to fish (doubled) (PR_N x n)
            channel_angles_surrounding = self.channel_angles_surrounding_2 + fish_angle
            self.red_readings = self._read(masked_arena_pixels[:, :, 0:1], eye_x, eye_y, channel_angles_surrounding, self.red_photoreceptor_num)

            if self.red_photoreceptor_num != self.uv_photoreceptor_num:
                self.pad_observation()
            else:
                self.readings = self.chosen_math_library.concatenate((self.uv_readings, self.red_readings), axis=1)

    def _read(self, masked_arena_pixels, eye_x, eye_y, channel_angles_surrounding, n_channels):
        """Lines method to return pixel sum for all points for each photoreceptor, over its segment."""

        # Make sure is in desired range (PR_N x n)
        channel_angles_surrounding_scaling = (channel_angles_surrounding // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        channel_angles_surrounding = channel_angles_surrounding + channel_angles_surrounding_scaling

        # Compute m using tan (PR_N x n)
        m = self.chosen_math_library.tan(channel_angles_surrounding)

        # Compute c (PR_N x n)
        c = -m * eye_x
        c = c + eye_y

        # Compute components of intersections (PR_N x n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 2)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 2)

        m_mul = self.chosen_math_library.expand_dims(m, 2)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 2)
        m_mul = full_m * self.mul1_full[:n_channels]
        m_mul[:, :, :3] = 1
        addition_matrix = self.addition_matrix[:n_channels] * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c_exp * self.multiplication_matrix[:n_channels]) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[:n_channels]) + self.add_for_hypothetical[:n_channels]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:n_channels]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates, (n_channels, self.n, 2, 2))
        # Get intersections (PR_N x 2)
        eye_position = self.chosen_math_library.array([eye_x, eye_y])
        possible_vectors = valid_intersection_coordinates - eye_position

        angles = self.chosen_math_library.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range.
        angle_scaling = (angles // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        angles = angles + angle_scaling

        angles = self.chosen_math_library.round(angles, 3)
        channel_angles_surrounding = self.chosen_math_library.round(channel_angles_surrounding, 3)

        channel_angles_surrounding = self.chosen_math_library.expand_dims(channel_angles_surrounding, 2)
        channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, 2, 2)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = self.chosen_math_library.reshape(selected_intersections, (n_channels, self.n, 1, 2))

        eye_position_full = self.chosen_math_library.tile(eye_position, (n_channels, self.n, 1, 1))
        vertices = self.chosen_math_library.concatenate((eye_position_full, selected_intersections), axis=2)
        vertices_xvals = vertices[:, :, :, 0]
        vertices_yvals = vertices[:, :, :, 1]

        min_x = self.chosen_math_library.min(vertices_xvals, axis=2)
        max_x = self.chosen_math_library.max(vertices_xvals, axis=2)
        min_y = self.chosen_math_library.min(vertices_yvals, axis=2)
        max_y = self.chosen_math_library.max(vertices_yvals, axis=2)

        # SEGMENT COMPUTATION
        x_lens = self.chosen_math_library.rint(max_x[:, 0] - min_x[:, 0])
        y_lens = self.chosen_math_library.rint(max_y[:, 0] - min_y[:, 0])

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

        full_set = full_set.swapaxes(0, 1)
        full_set = full_set.reshape(n_channels, -1, 2)

        masked_arena_pixels = masked_arena_pixels[full_set[:, :, 1], full_set[:, :, 0]]  # NOTE: Inverting x and y to match standard in program.
        total_sum = masked_arena_pixels.sum(axis=1)

        # Compute oversampling ratio. This takes account of how many indexes have been computed for each sector, and
        # scales all by this so there is an even density of pixel counts (otherwise small rays would be counted more).
        # (PR_N)
        oversampling_ratio = (x_lens + y_lens)/(x_len + y_len)
        oversampling_ratio = self.chosen_math_library.expand_dims(oversampling_ratio, 1)
        oversampling_ratio = self.chosen_math_library.repeat(oversampling_ratio, total_sum.shape[1], 1)
        total_sum = total_sum * (oversampling_ratio/self.n)

        total_sum = total_sum / 3

        return total_sum

    def get_sector_vertices(self, eye_x, eye_y, fish_angle):
        # TODO: Make suitable for different sets of photoreceptors
        """Uses lines method to return the vertices of all photoreceptor segments."""
        # Angles with respect to fish (doubled) (PR_N x n)
        channel_angles_surrounding = self.channel_angles_surrounding + fish_angle

        # Make sure is in desired range (PR_N x n) TODO: might need to find way of doing it multiple times e.g. by // operation
        below_range = (channel_angles_surrounding < 0) * self.chosen_math_library.pi * 2
        channel_angles_surrounding = channel_angles_surrounding + below_range
        above_range = (channel_angles_surrounding > self.chosen_math_library.pi * 2) * -self.chosen_math_library.pi * 2
        channel_angles_surrounding = channel_angles_surrounding + above_range

        # Compute m using tan (PR_N x n)
        m = self.chosen_math_library.tan(channel_angles_surrounding)

        # Compute c (PR_N x n)
        c = -m * eye_x
        c = c + eye_y

        # Compute components of intersections (PR_N x n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 2)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 2)

        m_mul = self.chosen_math_library.expand_dims(m, 2)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 2)
        m_mul = full_m * self.mul1_full
        m_mul[:, :, :3] = 1
        addition_matrix = self.addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c_exp * self.multiplication_matrix) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical) + self.add_for_hypothetical

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates, (self.photoreceptor_num, self.n, 2, 2))

        # Get intersections (PR_N x 2)
        eye_position = self.chosen_math_library.array([eye_x, eye_y])
        possible_vectors = valid_intersection_coordinates - eye_position

        angles = self.chosen_math_library.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        below_range = (angles < 0) * self.chosen_math_library.pi * 2
        angles = angles + below_range
        above_range = (angles > self.chosen_math_library.pi * 2) * -self.chosen_math_library.pi * 2
        angles = angles + above_range

        angles = self.chosen_math_library.round(angles, 2)
        channel_angles_surrounding = self.chosen_math_library.round(channel_angles_surrounding, 2)

        channel_angles_surrounding = self.chosen_math_library.expand_dims(channel_angles_surrounding, 2)
        channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, 2, 2)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = self.chosen_math_library.reshape(selected_intersections, (self.photoreceptor_num, self.n, 1, 2))

        relevant_intersections_1 = selected_intersections[:, 0, :, :]
        relevant_intersections_2 = selected_intersections[:, self.n-1, :, :]

        eye_position_full = self.chosen_math_library.tile(eye_position, (self.photoreceptor_num, 1, 1))
        vertices = self.chosen_math_library.concatenate((eye_position_full, relevant_intersections_1, relevant_intersections_2), axis=1)

        if self.using_gpu:
            return vertices.get()
        else:
            return vertices

    def show_points(self):
        pass

    def compute_n(self, photoreceptor_rf_size, max_separation=1):
        max_dist = (self.width**2 + self.height**2)**0.5
        theta_separation = math.asin(max_separation/max_dist)
        n = (photoreceptor_rf_size/theta_separation)/2
        return int(n)

    def pad_observation(self):
        """Makes photoreceptor input from two sources the same dimension by padding out points with their nearest values."""
        # TODO: Problem with this version - increases input from certain areas (like having areas with greater emphasis)

        if self.uv_photoreceptor_num < self.red_photoreceptor_num:
            self.readings = self.chosen_math_library.concatenate((self.red_readings, self.uv_readings[self.indices_for_padding]), axis=1)
        else:
            self.readings = self.chosen_math_library.concatenate((self.red_readings[self.indices_for_padding], self.uv_readings), axis=1)




