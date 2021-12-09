import numpy as np
import cupy as cp
import math


class Eye:

    def __init__(self, board, verg_angle, retinal_field, is_left, num_arms, min_distance, max_distance, dark_gain,
                 light_gain, bkg_scatter, dark_col, photoreceptor_rf_size, using_gpu):
        self.num_arms = num_arms
        self.distances = np.array([min_distance, max_distance])

        self.vis_angles = None
        self.dist = None
        self.theta = None

        # Use CUPY if using GPU.
        self.using_gpu = using_gpu
        if using_gpu:
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

        self.update_angles(verg_angle, retinal_field, is_left)
        self.readings = self.chosen_math_library.zeros((num_arms, 3), 'int')
        self.board = board
        self.dark_gain = dark_gain
        self.light_gain = light_gain
        self.bkg_scatter = bkg_scatter
        self.dark_col = dark_col

        self.width, self.height = self.board.get_size()

        # TODO: Make parameters:
        self.photoreceptor_num = num_arms
        self.photoreceptor_rf_size = photoreceptor_rf_size
        self.retinal_field_size = retinal_field
        self.photoreceptor_spacing = self.retinal_field_size/self.photoreceptor_num

        self.n = self.compute_n()
        self.observation_scaling_factor = self.n * 30

        # Compute repeated measures:
        self.channel_angles_surrounding = None
        self.mul_for_hypothetical = None
        self.add_for_hypothetical = None
        self.mul1_full = None
        self.addition_matrix = None
        self.conditional_tiled = None
        self.multiplication_matrix = None
        self.get_repeated_computations()

    def get_repeated_computations(self):
        channel_angles_surrounding = self.chosen_math_library.expand_dims(self.vis_angles, 1)
        channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, self.n, 1)
        rf_offsets = self.chosen_math_library.linspace(-self.photoreceptor_rf_size / 2, self.photoreceptor_rf_size / 2, num=self.n)
        self.channel_angles_surrounding = channel_angles_surrounding + rf_offsets

        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical, (self.photoreceptor_num, self.n, 1, 1))
        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        self.add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical, (self.photoreceptor_num, self.n, 1, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        self.mul1_full = self.chosen_math_library.tile(mul1, (self.photoreceptor_num, self.n, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.height - 1, self.width - 1])
        self.addition_matrix = self.chosen_math_library.tile(addition_matrix_unit, (self.photoreceptor_num, self.n, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        self.conditional_tiled = self.chosen_math_library.tile(conditional_tiled, (self.photoreceptor_num, self.n, 1))

        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        self.multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit, (self.photoreceptor_num, self.n, 1))

    def update_angles(self, verg_angle, retinal_field, is_left):
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        self.vis_angles = self.chosen_math_library.linspace(min_angle, max_angle, self.num_arms)

    def read(self, masked_arena_pixels, eye_x, eye_y, fish_angle):
        # masked_arena_pixels = cp.array(masked_arena_pixels)  # TODO: Make cp to begin with.
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
        eye_position_full = self.chosen_math_library.tile(eye_position, (self.photoreceptor_num, self.n, 1, 1))
        vertices = self.chosen_math_library.concatenate((eye_position_full, selected_intersections), axis=2)
        vertices_xvals = vertices[:, :, :, 0]
        vertices_yvals = vertices[:, :, :, 1]

        # TODO: Probably faster way of doing below...
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
        full_set = full_set.reshape(self.photoreceptor_num, -1, 2)

        # grid = np.zeros((self.width, self.height))
        #
        # # For investigative purposes
        # for i in range(self.photoreceptor_num):
        #     unique, counts = np.unique(full_set[i, :, :].get(), axis=0, return_counts=True)
        #     # counts = np.expand_dims(counts, 1)
        #     # frequencies = np.concatenate((unique, counts), axis=1)
        #     grid[unique[:, 0], unique[:, 1]] = counts*oversampling_ratio[i].get()
        #     grid = grid/self.n

        # full_set = full_set.reshape(self.photoreceptor_num, -1, 2).astype(int)
        # grid = np.zeros((self.width, self.height))
        # grid[full_set[0, :, 0], full_set[0, :, 1]] = 1

        # indexes = cp.zeros((self.width, self.height, 3), dtype=int)

        # self.readings[:, 0] = (masked_arena_pixels[:, :, 0] * selected_points).sum(axis=1)
        # self.readings[:, 1] = (masked_arena_pixels[:, :, 1] * selected_points).sum(axis=1)
        # self.readings[:, 2] = (masked_arena_pixels[:, :, 2] * selected_points).sum(axis=1)

        # total_sum = cp.zeros((self.photoreceptor_num, 3))

        # # full_set = full_set.get()
        # for i in range(self.photoreceptor_num):
        #     indexes[full_set[i, :, 0], full_set[i, :, 1], :] = 1
        #     total_sum[i] = (masked_arena_pixels * indexes).sum(axis=(0, 1))
        #     # indexes = np.unique(full_set[i, :, :], axis=0)
        #     # grid[indexes[:, 0], indexes[:, 1]] = 1
        #     indexes[:, :, :] = 0
            # self.readings[i] = masked_arena_pixels[indexes[:, 0], indexes[:, 1]].sum(axis=0)
        masked_arena_pixels = masked_arena_pixels[full_set[:, :, 1], full_set[:, :, 0]]  # NOTE: Inverting x and y to match standard in program.
        total_sum = masked_arena_pixels.sum(axis=1)

        # Compute oversampling ratio. This takes account of how many indexes have been computed for each sector, and
        # scales all by this so there is an even density of pixel counts (otherwise small rays would be counted more).
        # (PR_N)
        oversampling_ratio = (x_lens + y_lens)/(x_len + y_len)
        oversampling_ratio = self.chosen_math_library.expand_dims(oversampling_ratio, 1)
        oversampling_ratio = self.chosen_math_library.repeat(oversampling_ratio, 3, 1)
        total_sum = total_sum * (oversampling_ratio/self.n)

        total_sum = total_sum / 3

        self.readings = total_sum

    def get_sector_vertices(self, eye_x, eye_y, fish_angle):
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

    def compute_n(self, max_separation=1):
        max_dist = (self.width**2 + self.height**2)**0.5
        theta_separation = math.asin(max_separation/max_dist)
        n = self.photoreceptor_rf_size/theta_separation
        return int(n)  # TODO: Change to appropriate value.
