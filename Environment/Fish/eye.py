import numpy as np
import cupy as cp
import math
from skimage.draw import line
from skimage.transform import resize
import matplotlib.pyplot as plt

# from Tools.resize import resize

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
        self.isomerization_probability = env_variables['isomerization_frequency']/env_variables['sim_steps_per_second']
        self.dark_col = dark_col
        self.dist = None
        self.theta = None
        self.width, self.height = self.board.get_size()
        self.retinal_field_size = retinal_field
        self.env_variables = env_variables

        # To allow padding of channel observations if are of different size

        if env_variables['shared_photoreceptor_channels']:
            if env_variables['uv_photoreceptor_num'] != env_variables['red_photoreceptor_num']:
                print("Set numbers of UV and Red channels dont match, using UV channel number for both.")
            self.shared_photoreceptor_channels = True

            self.photoreceptor_num = env_variables['uv_photoreceptor_num']
            self.photoreceptor_rf_size = env_variables['uv_photoreceptor_rf_size']
            self.photoreceptor_spacing = self.retinal_field_size / self.photoreceptor_num
            self.max_photoreceptor_num = self.photoreceptor_num

            if env_variables['incorporate_uv_strike_zone']:
                self.photoreceptor_angles = self.update_angles_strike_zone(verg_angle, retinal_field, is_left,
                                                                           self.photoreceptor_num,
                                                                           env_variables["strike_zone_sigma"])
            else:
                self.photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left,
                                                               self.photoreceptor_num)
            self.readings = self.chosen_math_library.zeros((self.photoreceptor_num, 3),
                                                           'int')  # Three dimensions: Red, UV, and background (red)

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
            self.total_photoreceptor_num = self.uv_photoreceptor_num + self.red_photoreceptor_num

            if env_variables['incorporate_uv_strike_zone']:
                self.uv_photoreceptor_angles = self.update_angles_strike_zone(verg_angle, retinal_field, is_left,
                                                                              self.uv_photoreceptor_num,
                                                                              env_variables["strike_zone_sigma"])
            else:
                self.uv_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left,
                                                                  self.uv_photoreceptor_num, True)
            self.red_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left,
                                                               self.red_photoreceptor_num)

            self.uv_readings = self.chosen_math_library.zeros((env_variables['uv_photoreceptor_num'], 1), 'int')
            self.red_readings = self.chosen_math_library.zeros((env_variables['red_photoreceptor_num'], 1), 'int')

            # Compute minimum lines that need to be extrapolated so as not to miss coordinates.
            self.n = self.compute_n(max([self.uv_photoreceptor_rf_size, self.red_photoreceptor_rf_size]))

            self.indices_for_padding = self.chosen_math_library.around(
                self.chosen_math_library.linspace(0, self.min_photoreceptor_num - 1,
                                                  self.max_photoreceptor_num)).astype(
                int)

        # Decide whether to pad observation to larger factor than either size (avoids conv layer bug)
        if self.max_photoreceptor_num < self.env_variables['minimum_observation_size']:
            self.expand_observation = True
            self.observation_size = self.env_variables['minimum_observation_size']
        else:
            self.expand_observation = False
            self.observation_size = self.max_photoreceptor_num

        # Compute repeated measures:
        self.channel_angles_surrounding = None
        self.channel_angles_surrounding_red = None
        self.mul_for_hypothetical = None
        self.add_for_hypothetical = None
        self.mul1_full = None
        self.addition_matrix = None
        self.conditional_tiled = None
        self.multiplication_matrix = None
        self.get_repeated_computations()

        self.channel_angles_surrounding_stacked = self.chosen_math_library.concatenate((self.channel_angles_surrounding,
                                                                                        self.channel_angles_surrounding_red), axis=0)

        self.snr = []
        self.red_signal_fail = []
        self.uv_signal_fail = []
        self.red2_signal_fail = []

    def get_repeated_computations(self):
        if self.shared_photoreceptor_channels:
            channel_angles_surrounding = self.chosen_math_library.expand_dims(self.photoreceptor_angles, 1)
            channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, self.n, 1)
            rf_offsets = self.chosen_math_library.linspace(-self.photoreceptor_rf_size / 2,
                                                           self.photoreceptor_rf_size / 2, num=self.n)
            self.channel_angles_surrounding = channel_angles_surrounding + rf_offsets

            n_channels_in_computation_axis_0 = self.max_photoreceptor_num
        else:
            # UV
            channel_angles_surrounding = self.chosen_math_library.expand_dims(self.uv_photoreceptor_angles, 1)
            channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, self.n, 1)
            rf_offsets = self.chosen_math_library.linspace(-self.uv_photoreceptor_rf_size / 2,
                                                           self.uv_photoreceptor_rf_size / 2, num=self.n)
            self.channel_angles_surrounding = channel_angles_surrounding + rf_offsets

            # Red
            channel_angles_surrounding_2 = self.chosen_math_library.expand_dims(self.red_photoreceptor_angles, 1)
            channel_angles_surrounding_2 = self.chosen_math_library.repeat(channel_angles_surrounding_2, self.n, 1)
            rf_offsets_2 = self.chosen_math_library.linspace(-self.red_photoreceptor_rf_size / 2,
                                                             self.red_photoreceptor_rf_size / 2, num=self.n)
            self.channel_angles_surrounding_red = channel_angles_surrounding_2 + rf_offsets_2

            n_channels_in_computation_axis_0 = self.total_photoreceptor_num

        # Same for both, just requires different dimensions
        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical,
                                                                  (n_channels_in_computation_axis_0, self.n, 1, 1))
        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        self.add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical,
                                                                  (n_channels_in_computation_axis_0, self.n, 1, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        self.mul1_full = self.chosen_math_library.tile(mul1, (n_channels_in_computation_axis_0, self.n, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.height - 1, self.width - 1])
        self.addition_matrix = self.chosen_math_library.tile(addition_matrix_unit,
                                                             (n_channels_in_computation_axis_0, self.n, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        self.conditional_tiled = self.chosen_math_library.tile(conditional_tiled,
                                                               (n_channels_in_computation_axis_0, self.n, 1))

        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        self.multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit,
                                                                   (n_channels_in_computation_axis_0, self.n, 1))

    def update_angles(self, verg_angle, retinal_field, is_left, photoreceptor_num, is_uv=False):
        """Set the eyes visual angles to be an even distribution."""
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

    def create_half_normal_distribution(self, min_angle, max_angle, photoreceptor_num, sigma=1):
        mu = 0
        angle_difference = abs(max_angle - min_angle)

        angle_range = np.linspace(min_angle, max_angle, photoreceptor_num)
        frequencies = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(angle_range - mu) ** 2 / (2 * sigma ** 2))
        differences = 1 / frequencies
        differences[0] = 0
        total_difference = np.sum(differences)
        differences = (differences * angle_difference) / total_difference
        cumulative_differences = np.cumsum(differences)
        photoreceptor_angles = min_angle + cumulative_differences

        # Computing Indices for resampling
        # hypothetical_angle_range = np.linspace(min_angle, max_angle, self.max_photoreceptor_num)
        # hypothetical_frequencies = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(hypothetical_angle_range-mu)**2/(2*sigma**2))
        # hypothetical_differences = 1/hypothetical_frequencies
        # hypothetical_differences[0] = 0
        # hypothetical_total_difference = np.sum(hypothetical_differences)
        # hypothetical_differences = (hypothetical_differences*angle_difference)/hypothetical_total_difference
        # hypothetical_cumulative_differences = np.cumsum(hypothetical_differences)
        # hypothetical_photoreceptor_angles = min_angle + hypothetical_cumulative_differences
        # relative_indices = np.round(((hypothetical_photoreceptor_angles - min(hypothetical_photoreceptor_angles))/angle_difference) * (photoreceptor_num-1)).astype(int)

        return photoreceptor_angles  # , relative_indices

    def update_angles_strike_zone(self, verg_angle, retinal_field, is_left, photoreceptor_num, sigma):
        """Set the eyes visual angles, with the option of particular distributions."""

        # Half normal distribution
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2

        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2

        # sampled_values = self.sample_half_normal_distribution(min_angle, max_angle, photoreceptor_num)
        computed_values = self.create_half_normal_distribution(min_angle, max_angle, photoreceptor_num, sigma)

        # plt.scatter(computed_values, computed_values)
        # plt.show()
        # self.indices_for_padding_uv = relative_indices

        return self.chosen_math_library.array(computed_values)

    def read(self, masked_arena_pixels, eye_x, eye_y, fish_angle):
        """
        Resolve RF coordinates for each photoreceptor, and use those to sum the relevant pixels.

        Does all in parallel if channels have same angles and sizes, otherwise runs them separately.
        """
        if self.shared_photoreceptor_channels:
            # Angles with respect to fish (doubled) (PR_N x n)
            channel_angles_surrounding = self.channel_angles_surrounding + fish_angle
            readings = self._read(masked_arena_pixels, eye_x, eye_y, channel_angles_surrounding, self.photoreceptor_num)
            scaled_readings = self.scale_readings(readings,
                                                  self.env_variables['red_scaling_factor'] * self.env_variables["light_gain"]/200,
                                                  self.env_variables['uv_scaling_factor'] * self.env_variables["light_gain"]/200,
                                                  self.env_variables['red_2_scaling_factor'] * self.env_variables["light_gain"]/200
)
            self.readings = self.add_noise_to_readings(scaled_readings)
            self.readings = scaled_readings

        else:
            # UV Angles with respect to fish (doubled) (PR_N x n)
            # channel_angles_surrounding = self.channel_angles_surrounding + fish_angle
            # uv_readings = self._read(masked_arena_pixels[:, :, 1:2], eye_x, eye_y, channel_angles_surrounding,
            #                          self.uv_photoreceptor_num)
            # uv_readings_scaled = self.scale_readings(uv_readings,
            #                                          self.env_variables['uv_scaling_factor'])
            # self.uv_readings = self.add_noise_to_readings(uv_readings_scaled)
            #
            # # Red Angles with respect to fish (for both field and background pixels) (PR_N x n)
            # channel_angles_surrounding = self.channel_angles_surrounding_red + fish_angle
            # red_arena_pixels = self.chosen_math_library.concatenate(
            #     (masked_arena_pixels[:, :, 0:1], masked_arena_pixels[:, :, 2:]), axis=2)
            # red_readings = self._read(red_arena_pixels, eye_x, eye_y, channel_angles_surrounding,
            #                           self.red_photoreceptor_num)
            # red_readings_scaled = self.scale_readings(red_readings,
            #                                           self.env_variables['red_scaling_factor'],
            #                                           self.env_variables['red_2_scaling_factor'])
            # self.red_readings = self.add_noise_to_readings(red_readings_scaled)

            # Chaned method so is now stacked.
            channel_angles_surrounding = self.channel_angles_surrounding_stacked + fish_angle
            uv_arena_pixels = masked_arena_pixels[:, :, 1:2]
            red_arena_pixels = self.chosen_math_library.concatenate(
                 (masked_arena_pixels[:, :, 0:1], masked_arena_pixels[:, :, 2:]), axis=2)
            uv_readings, red_readings = self._read_stacked(masked_arena_pixels_uv=uv_arena_pixels,
                                                           masked_arena_pixels_red=red_arena_pixels,
                                                           eye_x=eye_x,
                                                           eye_y=eye_y,
                                                           channel_angles_surrounding=channel_angles_surrounding,
                                                           n_channels_uv=self.uv_photoreceptor_num,
                                                           n_channels_red=self.red_photoreceptor_num)

            uv_readings_scaled = self.scale_readings(uv_readings,
                                                     self.env_variables['uv_scaling_factor'] * self.env_variables["light_gain"]/200)
            self.uv_readings = self.add_noise_to_readings(uv_readings_scaled)

            red_readings_scaled = self.scale_readings(red_readings,
                                                      self.env_variables['red_scaling_factor'] * self.env_variables["light_gain"]/200,
                                                      self.env_variables['red_2_scaling_factor'] * self.env_variables["light_gain"]/200)
            self.red_readings = self.add_noise_to_readings(red_readings_scaled)

            if self.red_photoreceptor_num != self.uv_photoreceptor_num:
                self.pad_observation()
            else:
                self.readings = self.chosen_math_library.concatenate(
                    (self.red_readings[:, 0:1], self.uv_readings, self.red_readings[:, 1:]), axis=1)

    def _read(self, masked_arena_pixels, eye_x, eye_y, channel_angles_surrounding, n_channels):
        """Lines method to return pixel sum for all points for each photoreceptor, over its segment."""

        # Make sure angles are in desired range (PR_N x n)
        channel_angles_surrounding_scaling = (channel_angles_surrounding // (
                    self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
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

        intersection_components = ((c_exp * self.multiplication_matrix[
                                            :n_channels]) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[
                                                               :n_channels]) + self.add_for_hypothetical[:n_channels]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:n_channels]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates,
                                                                          (n_channels, self.n, 2, 2))
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

        masked_arena_pixels = masked_arena_pixels[
            full_set[:, :, 1], full_set[:, :, 0]]  # NOTE: Inverting x and y to match standard in program.
        total_sum = masked_arena_pixels.sum(axis=1)

        # Compute oversampling ratio. This takes account of how many indexes have been computed for each sector, and
        # scales all by this so there is an even density of pixel counts (otherwise small rays would be counted more).
        # (PR_N)
        oversampling_ratio = (x_lens + y_lens) / (x_len + y_len)
        oversampling_ratio = self.chosen_math_library.expand_dims(oversampling_ratio, 1)
        oversampling_ratio = self.chosen_math_library.repeat(oversampling_ratio, total_sum.shape[1], 1)
        total_sum = total_sum * (oversampling_ratio / self.n)

        return total_sum

    def _read_stacked(self, masked_arena_pixels_uv, masked_arena_pixels_red, eye_x, eye_y, channel_angles_surrounding,
                      n_channels_uv, n_channels_red):
        """
        Lines method to return pixel sum for all points for each photoreceptor, over its segment.
        Desired performance improvement from stacking the channels along 0th axis so one call required rather than two.
        Takes channel_angles_surrounding with UV then Red
        """
        n_channels = n_channels_uv + n_channels_red

        # Make sure angles are in desired range (PR_N x n)
        channel_angles_surrounding_scaling = (channel_angles_surrounding // (
                    self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
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

        intersection_components = ((c_exp * self.multiplication_matrix[
                                            :n_channels]) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[
                                                               :n_channels]) + self.add_for_hypothetical[:n_channels]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:n_channels]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates,
                                                                          (n_channels, self.n, 2, 2))
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

        full_set_uv = full_set[:n_channels_uv, :]
        full_set_red = full_set[n_channels_uv:, :]

        # masked_arena_pixels = masked_arena_pixels[
        #     full_set[:, :, 1], full_set[:, :, 0]]  # NOTE: Inverting x and y to match standard in program.
        # total_sum = masked_arena_pixels.sum(axis=1)

        masked_arena_pixels_uv = masked_arena_pixels_uv[full_set_uv[:, :, 1], full_set_uv[:, :, 0]]  # NOTE: Inverting x and y to match standard in program. TODO: THIS IS WHERE BREAKS.
        total_sum_uv = masked_arena_pixels_uv.sum(axis=1)

        masked_arena_pixels_red = masked_arena_pixels_red[
            full_set_red[:, :, 1], full_set_red[:, :, 0]]  # NOTE: Inverting x and y to match standard in program.
        total_sum_red = masked_arena_pixels_red.sum(axis=1)

        # Compute oversampling ratio. This takes account of how many indexes have been computed for each sector, and
        # scales all by this so there is an even density of pixel counts (otherwise small rays would be counted more).
        # (PR_N)
        oversampling_ratio = (x_lens + y_lens) / (x_len + y_len)
        oversampling_ratio = self.chosen_math_library.expand_dims(oversampling_ratio, 1)
        oversampling_ratio_red = self.chosen_math_library.repeat(oversampling_ratio, total_sum_red.shape[1], 1)

        total_sum_uv = total_sum_uv * (oversampling_ratio[:n_channels_uv] / self.n)
        total_sum_red = total_sum_red * (oversampling_ratio_red[n_channels_uv:] / self.n)

        return total_sum_uv, total_sum_red

    @staticmethod
    def scale_readings(readings, readings_scaling_factor, readings_scaling_factor_2=False,
                       readings_scaling_factor_3=False):
        """
        Takes all readings, and scales them by the appropriate scaling factor (which is a result of distance
        calibration to set values between 0 and 1.
        """
        readings[:, 0] *= readings_scaling_factor
        if readings_scaling_factor_2:
            readings[:, 1] *= readings_scaling_factor_2
            if readings_scaling_factor_3:
                readings[:, 2] *= readings_scaling_factor_3
        # print(np.max(readings, axis=0))

        # readings[readings > 1] = 1
        return readings

    def add_noise_to_readings(self, readings):
        """As specified, adds shot, read, and/or dark noise to readings."""
        if self.env_variables["shot_noise"]:
            photons = self.chosen_math_library.random.poisson(readings)
            shot_noise_difference = self.chosen_math_library.abs(readings - photons)
            snr = 1 - np.mean(shot_noise_difference/(photons+1), axis=1)
        else:
            photons = readings

        if self.env_variables['read_noise_sigma'] > 0:
            noise = self.chosen_math_library.random.randn(readings.shape[0], readings.shape[1]) * self.env_variables['read_noise_sigma']
            photons += noise.astype(int)
            photons = self.chosen_math_library.clip(photons, 0, 255)

        # if photons.shape[1] == 1:
        #     print(f"Max photons: {np.max(readings[:, 0])}")
        #
        #     self.uv_signal_fail.append(snr)
        # else:
        #
        #     self.red_signal_fail.append(snr[0])
        #
        #
        #     self.red2_signal_fail.append(snr[1])


        if self.env_variables["isomerization_frequency"] > 0:
            dark_noise_events = self.chosen_math_library.random.choice([0, 1], size=readings.size,
                                                                       p=[1-self.isomerization_probability,
                                                                          self.isomerization_probability]
                                                                       ).astype(float)
            variability = self.chosen_math_library.random.rand(readings.size)
            dark_noise_events *= variability
            dark_noise_events = dark_noise_events * self.env_variables['max_isomerization_size']
            dark_noise_events = self.chosen_math_library.reshape(dark_noise_events, readings.shape)
            photons += dark_noise_events

        return photons

    def get_sector_vertices(self, eye_x, eye_y, channel_angles_surrounding, n_channels):
        """Uses lines method to return the vertices of all photoreceptor segments."""
        # Make sure is in desired range (PR_N x n)
        channel_angles_surrounding_scaling = (channel_angles_surrounding // (
                    self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
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

        intersection_components = ((c_exp * self.multiplication_matrix[
                                            :n_channels]) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[
                                                               :n_channels]) + self.add_for_hypothetical[:n_channels]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:n_channels]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates,
                                                                          (n_channels, self.n, 2, 2))

        # Get intersections (PR_N x 2)
        eye_position = self.chosen_math_library.array([eye_x, eye_y])
        possible_vectors = valid_intersection_coordinates - eye_position

        angles = self.chosen_math_library.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure is in desired range (PR_N x n)
        angles_scaling = (angles // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        angles = angles + angles_scaling

        angles = self.chosen_math_library.round(angles, 2)
        channel_angles_surrounding = self.chosen_math_library.round(channel_angles_surrounding, 2)

        channel_angles_surrounding = self.chosen_math_library.expand_dims(channel_angles_surrounding, 2)
        channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, 2, 2)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = self.chosen_math_library.reshape(selected_intersections,
                                                                  (n_channels, self.n, 1, 2))
        # try:
        #     selected_intersections = self.chosen_math_library.reshape(selected_intersections,
        #                                                               (n_channels, self.n, 1, 2))
        # except ValueError:
        #     print(f"Angles: {angles}")
        #     print(f"Channel angles durrounding: {channel_angles_surrounding}")
        #     print(f"Eye position: {eye_position}")
        #     print(f"Fish angle: {fish_angle}")
        #     print(f"Number of channels: {n_channels}")

        relevant_intersections_1 = selected_intersections[:, 0, :, :]
        relevant_intersections_2 = selected_intersections[:, self.n - 1, :, :]

        eye_position_full = self.chosen_math_library.tile(eye_position, (n_channels, 1, 1))
        vertices = self.chosen_math_library.concatenate(
            (eye_position_full, relevant_intersections_1, relevant_intersections_2), axis=1)

        if self.using_gpu:
            return vertices.get()
        else:
            return vertices

    def show_points(self, bx, by, fish_angle):
        """Displays relevant parts of visual system, including (optional):
        * Channel sectors
        * Obstruction mask
        * Scatter mask
        * Light mask
        """
        # UV
        channel_angles_surrounding = self.channel_angles_surrounding + fish_angle
        sector_vertices = self.get_sector_vertices(bx, by, channel_angles_surrounding, self.uv_photoreceptor_num)
        sector_vertices = sector_vertices.astype(int)

        for sector in sector_vertices:
            [rr, cc] = line(sector[0][1], sector[0][0], sector[1][1], sector[1][0])
            self.board.db[rr, cc] = (0, 0, 0.5)

            [rr, cc] = line(sector[1][1], sector[1][0], sector[2][1], sector[2][0])
            self.board.db[rr, cc] = (0, 0, 0.5)

            if len(sector) > 3:
                [rr, cc] = line(sector[2][1], sector[2][0], sector[3][1], sector[3][0])
                self.board.db[rr, cc] = (0, 0, 0.5)
                [rr, cc] = line(sector[3][1], sector[3][0], sector[0][1], sector[0][0])
            else:
                [rr, cc] = line(sector[2][1], sector[2][0], sector[0][1], sector[0][0])
            self.board.db[rr, cc] = (0, 0, 0.5)

        # Red
        channel_angles_surrounding = self.channel_angles_surrounding_red + fish_angle
        sector_vertices = self.get_sector_vertices(bx, by, channel_angles_surrounding, self.red_photoreceptor_num)
        sector_vertices = sector_vertices.astype(int)

        for sector in sector_vertices:
            [rr, cc] = line(sector[0][1], sector[0][0], sector[1][1], sector[1][0])
            self.board.db[rr, cc] = (0.5, 0, 0)

            [rr, cc] = line(sector[1][1], sector[1][0], sector[2][1], sector[2][0])
            self.board.db[rr, cc] = (0.5, 0, 0)

            if len(sector) > 3:
                [rr, cc] = line(sector[2][1], sector[2][0], sector[3][1], sector[3][0])
                self.board.db[rr, cc] = (0.5, 0, 0)
                [rr, cc] = line(sector[3][1], sector[3][0], sector[0][1], sector[0][0])
            else:
                [rr, cc] = line(sector[2][1], sector[2][0], sector[0][1], sector[0][0])
            self.board.db[rr, cc] = (0.5, 0, 0)

    def compute_n(self, photoreceptor_rf_size, max_separation=1):
        max_dist = (self.width ** 2 + self.height ** 2) ** 0.5
        theta_separation = math.asin(max_separation / max_dist)
        n = (photoreceptor_rf_size / theta_separation)# / 2
        return int(n)

    def pad_observation(self):
        """Makes photoreceptor input from two sources the same dimension by padding out points with their nearest values."""
        # TODO: Note the switching below might be slow... Probably worth writing own function.
        if self.expand_observation:
            # Test
            if self.using_gpu:
                uv_readings = self.uv_readings.get()
                red_readings = self.red_readings.get()

            else:
                uv_readings = self.uv_readings
                red_readings = self.red_readings

            uv_readings = uv_readings.astype(float)
            red_readings = red_readings.astype(float)

            red_field = self.chosen_math_library.array(resize(red_readings[:, 0:1], (self.env_variables['minimum_observation_size'], 1), self.chosen_math_library))
            uv = self.chosen_math_library.array(resize(uv_readings, (self.env_variables['minimum_observation_size'], 1), self.chosen_math_library))
            red_background = self.chosen_math_library.array(resize(red_readings[:, 1:], (self.env_variables['minimum_observation_size'], 1), self.chosen_math_library))
            self.readings = self.chosen_math_library.concatenate((red_field, uv, red_background), axis=1)

        else:
            if self.uv_photoreceptor_num < self.red_photoreceptor_num:
                # Using linear interpolation
                # self.readings = self.chosen_math_library.concatenate((self.red_readings, self.uv_readings[self.indices_for_padding]), axis=1)

                if self.using_gpu:
                    uv_readings = self.uv_readings.get()
                else:
                    uv_readings = self.uv_readings
                self.readings = self.chosen_math_library.concatenate((self.red_readings[:, 0:1],
                                                                      self.chosen_math_library.array(resize(uv_readings,
                                                                                                            (
                                                                                                            self.red_photoreceptor_num,
                                                                                                            1))),
                                                                      self.red_readings[:, 1:]), axis=1)
            else:
                if self.using_gpu:
                    red_readings = self.red_readings.get()
                else:
                    red_readings = self.red_readings
                red_field = self.chosen_math_library.array(resize(red_readings[:, 0:1], (self.uv_photoreceptor_num, 1)))
                red_background = self.chosen_math_library.array(resize(red_readings[:, 1:], (self.uv_photoreceptor_num, 1)))
                self.readings = self.chosen_math_library.concatenate((red_field, self.uv_readings, red_background), axis=1)
                # self.readings = self.chosen_math_library.concatenate((self.red_readings[self.indices_for_padding], self.uv_readings), axis=1)

        # Used for debugging:
        # empty_dim = np.zeros((1, 120, 1))
        #
        # if self.using_gpu:
        #     observation = self.readings.get()
        # else:
        #     observation = self.readings
        # left_1 = observation[:, :] * 255
        #
        # left_1 = np.expand_dims(left_1, 0)
        # left_1 = np.concatenate((left_1[:, :, :1], empty_dim, left_1[:, :, 1:]), axis=2)
        # fig, axs = plt.subplots(2, 1, sharex=True)
        #
        # if self.using_gpu:
        #     observation = readings_2.get()
        # else:
        #     observation = readings_2
        # left_2 = observation[:, :] * 255
        #
        # left_2 = np.expand_dims(left_2, 0)
        # left_2 = np.concatenate((left_2[:, :, :1], empty_dim, left_2[:, :, 1:]), axis=2)
        #
        #
        # axs[0].imshow(left_1, aspect="auto")
        # axs[0].set_ylabel("Left eye")
        # axs[1].imshow(left_2, aspect="auto")
        # axs[1].set_ylabel("Left eye")
        #
        # axs[1].set_xlabel("Photoreceptor")
        # plt.show()
        #
        # fig, axs = plt.subplots(2, 1, sharex=True)
        #
        # if self.using_gpu:
        #     observation = self.uv_readings.get()
        # else:
        #     observation = self.uv_readings
        # left_3 = observation[:, :] * 255
        #
        # empty_dim1 = np.zeros((1, 55, 1))
        #
        # left_3 = np.expand_dims(left_3, 0)
        # left_3 = np.concatenate((empty_dim1, empty_dim1, left_3[:, :, :]), axis=2)
        #
        # axs[1].imshow(left_3, aspect="auto")
        # axs[1].set_ylabel("Left eye")
        # plt.show()

    def get_pr_coverage(self, masked_arena_pixels_uv, masked_arena_pixels_red, eye_x, eye_y, channel_angles_surrounding,
                        n_channels_uv, n_channels_red):
        """For testing purposes"""
        n_channels = n_channels_uv + n_channels_red

        # Make sure angles are in desired range (PR_N x n)
        channel_angles_surrounding_scaling = (channel_angles_surrounding // (
                    self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
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

        intersection_components = ((c_exp * self.multiplication_matrix[
                                            :n_channels]) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[
                                                               :n_channels]) + self.add_for_hypothetical[:n_channels]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:n_channels]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates,
                                                                          (n_channels, self.n, 2, 2))
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

        full_set_uv = full_set[:n_channels_uv, :]
        full_set_red = full_set[n_channels_uv:, :]

        return full_set_uv, full_set_red

    def get_pr_line_coordinates_uv(self, eye_x, eye_y):
        """For testing purposes"""
        # Make sure angles are in desired range (PR_N x n)
        channel_angles_surrounding_scaling = (self.uv_photoreceptor_angles // (
                    self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        channel_angles_surrounding = self.uv_photoreceptor_angles + channel_angles_surrounding_scaling

        # Compute m using tan (PR_N x n)
        m = self.chosen_math_library.tan(channel_angles_surrounding)

        # Compute c (PR_N x n)
        c = -m * eye_x
        c = c + eye_y

        # Compute components of intersections (PR_N x n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 1)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 1)

        m_mul = self.chosen_math_library.expand_dims(m, 1)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 1)
        m_mul = full_m * self.mul1_full[:self.uv_photoreceptor_num, 0]
        m_mul[:, :3] = 1
        addition_matrix = self.addition_matrix[:self.uv_photoreceptor_num, 0] * m_mul
        division_matrix = full_m
        division_matrix[:, 1] = 1
        division_matrix[:, 3] = 1

        intersection_components = ((c_exp * self.multiplication_matrix[
                                            :self.uv_photoreceptor_num, 0]) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 2)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 2)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[
                                                               :self.uv_photoreceptor_num, 0]) + self.add_for_hypothetical[:self.uv_photoreceptor_num, 0]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:self.uv_photoreceptor_num, 0]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates,
                                                                          (self.uv_photoreceptor_num, 2, 2))
        # Get intersections (PR_N x 2)
        eye_position = self.chosen_math_library.array([eye_x, eye_y])
        possible_vectors = valid_intersection_coordinates - eye_position

        angles = self.chosen_math_library.arctan2(possible_vectors[:, :, 1], possible_vectors[:, :, 0])

        # Make sure angles are in correct range.
        angle_scaling = (angles // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        angles = angles + angle_scaling

        angles = self.chosen_math_library.round(angles, 3)
        channel_angles_surrounding = self.chosen_math_library.round(channel_angles_surrounding, 3)

        channel_angles_surrounding = self.chosen_math_library.expand_dims(channel_angles_surrounding, 1)
        channel_angles_surrounding = self.chosen_math_library.repeat(channel_angles_surrounding, 2, 1)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = self.chosen_math_library.reshape(selected_intersections, (self.uv_photoreceptor_num, 1, 2))

        eye_position_full = self.chosen_math_library.tile(eye_position, (self.uv_photoreceptor_num, 1, 1))
        vertices = self.chosen_math_library.concatenate((eye_position_full, selected_intersections), axis=1)
        vertices_xvals = vertices[:, :, 0]
        vertices_yvals = vertices[:, :, 1]

        min_x = self.chosen_math_library.min(vertices_xvals, axis=1)
        max_x = self.chosen_math_library.max(vertices_xvals, axis=1)
        min_y = self.chosen_math_library.min(vertices_yvals, axis=1)
        max_y = self.chosen_math_library.max(vertices_yvals, axis=1)

        # SEGMENT COMPUTATION
        x_lens = self.chosen_math_library.rint(max_x - min_x)
        y_lens = self.chosen_math_library.rint(max_y - min_y)

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
        full_set = full_set.reshape(self.uv_photoreceptor_num, -1, 2)

        return full_set
