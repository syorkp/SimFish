import numpy as np
import cupy as cp
import gc


class Eyes:

    def __init__(self, board, verg_angle, retinal_field, num_arms, min_distance, max_distance, dark_gain, light_gain,
                 bkg_scatter, dark_col):

        self.num_arms = num_arms
        self.distances = np.array([min_distance, max_distance])

        self.vis_angles_left, self.vis_angles_right = self.get_photoreceptor_angles(verg_angle, retinal_field)
        self.vis_angles = cp.concatenate((self.vis_angles_left, self.vis_angles_right), axis=0)
        self.dist = None
        self.theta = None

        self.readings = np.zeros((num_arms*2, 3), 'int')
        self.board = board
        self.dark_gain = dark_gain
        self.light_gain = light_gain
        self.bkg_scatter = bkg_scatter
        self.dark_col = dark_col

        self.width, self.height = self.board.get_size()

        # TODO: Make parameters:
        self.photoreceptor_num = num_arms * 2
        self.photoreceptor_num_per_eye = num_arms
        self.photoreceptor_rf_size = 0.014
        self.retinal_field_size = retinal_field
        self.photoreceptor_spacing = self.retinal_field_size/self.photoreceptor_num_per_eye

        # Compute repeated measures:
        self.channel_angles_surrounding = None
        self.mul_for_hypothetical = None
        self.add_for_hypothetical = None
        self.mul1_full = None
        self.addition_matrix = None
        self.conditional_tiled = None
        self.multiplication_matrix = None
        self.get_repeated_computations()

        # self.mempool = cp.get_default_memory_pool()

    def get_photoreceptor_angles(self, verg_angle, retinal_field):
        min_angle_left = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
        max_angle_left = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        vis_angles_left = cp.linspace(min_angle_left, max_angle_left, self.num_arms)

        min_angle_right = np.pi / 2 - retinal_field / 2 - verg_angle / 2
        max_angle_right = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        vis_angles_right = cp.linspace(min_angle_right, max_angle_right, self.num_arms)

        return vis_angles_left, vis_angles_right

    def get_repeated_computations(self, n=20):
        channel_angles_surrounding = cp.expand_dims(self.vis_angles, 1)
        channel_angles_surrounding = cp.repeat(channel_angles_surrounding, n, 1)
        rf_offsets = cp.linspace(-self.photoreceptor_rf_size / 2, self.photoreceptor_rf_size / 2, num=n)
        self.channel_angles_surrounding = channel_angles_surrounding + rf_offsets

        mul_for_hypothetical = cp.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = cp.tile(mul_for_hypothetical, (self.photoreceptor_num, n, 1, 1))
        add_for_hypothetical = cp.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        self.add_for_hypothetical = cp.tile(add_for_hypothetical, (self.photoreceptor_num, n, 1, 1))

        mul1 = cp.array([0, 0, 0, 1])
        self.mul1_full = cp.tile(mul1, (self.photoreceptor_num, n, 1))

        addition_matrix_unit = cp.array([0, 0, self.height - 1, self.width - 1])
        self.addition_matrix = cp.tile(addition_matrix_unit, (self.photoreceptor_num, n, 1))

        conditional_tiled = cp.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        self.conditional_tiled = cp.tile(conditional_tiled, (self.photoreceptor_num, n, 1))

        multiplication_matrix_unit = cp.array([-1, 1, -1, 1])
        self.multiplication_matrix = cp.tile(multiplication_matrix_unit, (self.photoreceptor_num, n, 1))

    def read(self, masked_arena_pixels, fish_angle, left_eye_pos, right_eye_pos, n=20):
        # right_eye_pos = (
        #     -np.cos(np.pi / 2 - fish_angle) * eyes_biasx + fish_x,
        #     +np.sin(np.pi / 2 - fish_angle) * eyes_biasx + fish_y)
        # left_eye_pos = (
        #     +np.cos(np.pi / 2 - fish_angle) * eyes_biasx + fish_x,
        #     -np.sin(np.pi / 2 - fish_angle) * eyes_biasx + fish_y)

        masked_arena_pixels = cp.array(masked_arena_pixels)

        # Angles with respect to fish (doubled) (PR_N x n)
        channel_angles_surrounding = self.channel_angles_surrounding + fish_angle

        # Make sure is in desired range (PR_N x n) TODO: might need to find way of doing it multiple times e.g. by // operation
        below_range = (channel_angles_surrounding < 0) * cp.pi * 2
        channel_angles_surrounding = channel_angles_surrounding + below_range
        above_range = (channel_angles_surrounding > cp.pi * 2) * -cp.pi * 2
        channel_angles_surrounding = channel_angles_surrounding + above_range

        # Compute m using tan (PR_N x n)
        m = cp.tan(channel_angles_surrounding)
        m_left, m_right = cp.split(m, 2)

        # Compute c (PR_N x n)
        c_left = -m_left * left_eye_pos[0]
        c_left = c_left + left_eye_pos[1]

        c_right = -m_right * right_eye_pos[0]
        c_right = c_right + right_eye_pos[1]

        c = cp.concatenate((c_left, c_right), axis=0)

        # Compute components of intersections (PR_N x n x 4)
        c_exp = cp.expand_dims(c, 2)
        c_exp = cp.repeat(c_exp, 4, 2)

        m_mul = cp.expand_dims(m, 2)
        full_m = cp.repeat(m_mul, 4, 2)
        m_mul = full_m * self.mul1_full
        m_mul[:, :, :3] = 1
        addition_matrix = self.addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c_exp * self.multiplication_matrix) + addition_matrix) / division_matrix

        intersection_coordinates = cp.expand_dims(intersection_components, 3)
        intersection_coordinates = cp.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical) + self.add_for_hypothetical

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = cp.reshape(valid_intersection_coordinates, (self.photoreceptor_num, n, 2, 2))

        # Split into streams for both eyes
        valid_intersection_coordinates_left, valid_intersection_coordinates_right = cp.split(valid_intersection_coordinates, 2)

        # Get intersections (PR_N x 2)
        left_eye_position = cp.array([left_eye_pos[0], left_eye_pos[1]])
        possible_vectors_left = valid_intersection_coordinates_left - left_eye_position

        right_eye_position = cp.array([right_eye_pos[0], right_eye_pos[1]])
        possible_vectors_right = valid_intersection_coordinates_right - right_eye_position

        possible_vectors = cp.concatenate((possible_vectors_left, possible_vectors_right), axis=0)

        angles = cp.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        below_range = (angles < 0) * cp.pi * 2
        angles = angles + below_range
        above_range = (angles > cp.pi * 2) * -cp.pi * 2
        angles = angles + above_range

        angles = cp.round(angles, 2)
        channel_angles_surrounding = cp.round(channel_angles_surrounding, 2)

        channel_angles_surrounding = cp.expand_dims(channel_angles_surrounding, 2)
        channel_angles_surrounding = cp.repeat(channel_angles_surrounding, 2, 2)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = cp.reshape(selected_intersections, (self.photoreceptor_num, n, 1, 2))

        eye_position_left = cp.tile(left_eye_position, (self.photoreceptor_num_per_eye, n, 1, 1))
        eye_position_right = cp.tile(right_eye_position, (self.photoreceptor_num_per_eye, n, 1, 1))
        eye_position_full = cp.concatenate((eye_position_left, eye_position_right), axis=0)
        vertices = cp.concatenate((eye_position_full, selected_intersections), axis=2)
        vertices_xvals = vertices[:, :, :, 0]
        vertices_yvals = vertices[:, :, :, 1]

        del above_range
        del addition_matrix
        del angles
        del below_range
        del c_exp
        del c_left
        del c_right
        del channel_angles_surrounding
        del division_matrix
        del eye_position_left
        del eye_position_right
        del eye_position_full
        del full_m
        del m_left
        del m_right
        del m_mul
        del valid_intersection_coordinates_left
        del valid_intersection_coordinates_right
        del valid_intersection_coordinates
        del same_values
        del selected_intersections
        del right_eye_position
        del left_eye_position
        del valid_points
        del valid_points_ls
        del valid_points_more
        del vertices
        gc.collect()
        cp._default_memory_pool.free_all_blocks()


        # TODO: Probably faster way of doing below...
        min_x = cp.min(vertices_xvals, axis=2)
        max_x = cp.max(vertices_xvals, axis=2)
        min_y = cp.min(vertices_yvals, axis=2)
        max_y = cp.max(vertices_yvals, axis=2)

        del vertices_xvals
        del vertices_yvals

        # SEGMENT COMPUTATION
        x_len = cp.max(np.rint(max_x[:, 0] - min_x[:, 0]).astype(int))
        y_len = cp.max(np.rint(max_y[:, 0] - min_y[:, 0]).astype(int))

        x_ranges = cp.linspace(min_x, max_x, int(x_len))
        y_ranges = cp.linspace(min_y, max_y, int(y_len))

        del min_x
        del max_x
        del min_y
        del max_y
        gc.collect()

        y_values = (m * x_ranges) + c
        y_values = cp.floor(y_values)
        set_1 = cp.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = cp.floor(x_values)
        set_2 = cp.stack((x_values, y_ranges), axis=-1)
        full_set = cp.vstack((set_1, set_2))

        del set_1
        del set_2
        del x_ranges
        del y_ranges
        del x_len
        del y_len
        del x_values
        del y_values
        gc.collect()

        cp._default_memory_pool.free_all_blocks()

        full_set = full_set.reshape(self.photoreceptor_num, -1, full_set.shape[-1]).astype(int)

        masked_arena_pixels = masked_arena_pixels[full_set[:, :, 0], full_set[:, :, 1]]
        total_sum = masked_arena_pixels.sum(axis=1)

        self.readings = total_sum.get()
        cp._default_memory_pool.free_all_blocks()
