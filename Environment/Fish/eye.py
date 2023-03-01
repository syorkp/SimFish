import numpy as np
import math
from skimage.draw import line
from skimage.transform import resize
import scipy.signal as signal
import matplotlib.pyplot as plt


class Eye:

    def __init__(self, board, verg_angle, retinal_field, is_left, env_variables, dark_col, using_gpu, max_visual_range,
                 plot_rfs=False):
        # Use CUPY if using GPU.
        self.using_gpu = using_gpu
        if using_gpu:
            import cupy as cp

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
        self.width, self.height = self.board.get_FOV_size()
        self.retinal_field_size = retinal_field
        self.env_variables = env_variables
        self.max_visual_range = max_visual_range
        self.prey_diam = self.env_variables['prey_size']

        if "sz_rf_spacing" in self.env_variables:
            self.sz_rf_spacing = self.env_variables["sz_rf_spacing"]
            self.sz_size = self.env_variables["sz_size"]
            self.sz_oversampling_factor = self.env_variables["sz_oversampling_factor"]
            self.sigmoid_steepness = self.env_variables["sigmoid_steepness"]
        else:  # TODO: KEPT FOR IMMEDIATE COMPATIBILITY, REMOVE LATER
            self.sz_rf_spacing = 0.04  # 2.3 deg.
            self.sz_size = 1.047  # 60 deg.
            self.sz_oversampling_factor = 2.5
            self.sigmoid_steepness = 5.0

        self.periphery_rf_spacing = self.sz_rf_spacing * self.sz_oversampling_factor
        self.density_range = self.periphery_rf_spacing - self.sz_rf_spacing

        self.ang_bin = 0.001  # this is the bin size for the projection
        self.ang = self.chosen_math_library.arange(-np.pi, np.pi + self.ang_bin,
                                                   self.ang_bin)  # this is the angle range for the projection

        self.filter_bins = 20  # this is the number of bins for the scattering filter

        self.uv_photoreceptor_rf_size = env_variables['uv_photoreceptor_rf_size']
        self.red_photoreceptor_rf_size = env_variables['red_photoreceptor_rf_size']

        self.uv_photoreceptor_angles = self.update_angles_sigmoid(verg_angle, retinal_field, is_left)
        self.uv_photoreceptor_num = len(self.uv_photoreceptor_angles)
        self.red_photoreceptor_num = self.uv_photoreceptor_num

        self.interpolated_observation = self.chosen_math_library.arange(
            self.chosen_math_library.min(self.uv_photoreceptor_angles),
            self.chosen_math_library.max(self.uv_photoreceptor_angles) + self.sz_rf_spacing / 2,
            self.sz_rf_spacing / 2)

        self.observation_size = len(self.interpolated_observation)

        self.red_photoreceptor_angles = self.update_angles(verg_angle, retinal_field, is_left,
                                                           self.red_photoreceptor_num)

        self.uv_readings = self.chosen_math_library.zeros((self.uv_photoreceptor_num, 1))
        self.red_readings = self.chosen_math_library.zeros((self.red_photoreceptor_num, 1))
        self.total_photoreceptor_num = self.uv_photoreceptor_num + self.red_photoreceptor_num
        # Compute minimum lines that need to be extrapolated so as not to miss coordinates.
        self.n = self.compute_n(max([self.uv_photoreceptor_rf_size, self.red_photoreceptor_rf_size]))

        if plot_rfs:
            self.plot_photoreceptors(self.uv_photoreceptor_angles.get(), self.red_photoreceptor_angles.get(),
                                     self.uv_photoreceptor_rf_size, self.red_photoreceptor_rf_size, is_left)
        # Compute repeated measures:
        self.photoreceptor_angles_surrounding = None
        self.photoreceptor_angles_surrounding_red = None
        self.mul_for_hypothetical = None
        self.add_for_hypothetical = None
        self.mul1_full = None
        self.addition_matrix = None
        self.conditional_tiled = None
        self.multiplication_matrix = None
        self.get_repeated_computations()

        self.photoreceptor_angles_surrounding_stacked = self.chosen_math_library.concatenate((self.photoreceptor_angles_surrounding,
                                                                                              self.photoreceptor_angles_surrounding_red),
                                                                                             axis=0)

    def plot_photoreceptors(self, uv_photoreceptor_angles, red_photoreceptor_angles, uv_photoreceptor_rf_size,
                            red_photoreceptor_rf_size, is_left):
        # Plot the photoreceptors on a polar plot:
        plt.ioff()

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_yticklabels([])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        uv_rfs = np.zeros((len(uv_photoreceptor_angles), 2))
        red_rfs = np.zeros((len(red_photoreceptor_angles), 2))
        uv_rfs[:, 0] = uv_photoreceptor_angles - uv_photoreceptor_rf_size / 2
        uv_rfs[:, 1] = uv_photoreceptor_angles + uv_photoreceptor_rf_size / 2
        red_rfs[:, 0] = red_photoreceptor_angles - red_photoreceptor_rf_size / 2
        red_rfs[:, 1] = red_photoreceptor_angles + red_photoreceptor_rf_size / 2
        r_uv = np.ones(uv_rfs.shape) * 0.9
        r_red = np.ones(red_rfs.shape) * 1.1
        ax.plot(uv_rfs.T, r_uv.T, color='b', alpha=0.3, linewidth=1)
        ax.plot(red_rfs.T, r_red.T, color='r', alpha=0.3, linewidth=1)
        if is_left:
            plt.savefig('left_eye.png')
        else:
            plt.savefig('right_eye.png')
        fig.clf()
        # plt.ion()

    def get_repeated_computations(self):

        # UV
        photoreceptor_angles_surrounding = self.chosen_math_library.expand_dims(self.uv_photoreceptor_angles, 1)
        photoreceptor_angles_surrounding = self.chosen_math_library.repeat(photoreceptor_angles_surrounding, self.n, 1)
        rf_offsets = self.chosen_math_library.linspace(-self.uv_photoreceptor_rf_size / 2,
                                                       self.uv_photoreceptor_rf_size / 2, num=self.n)
        self.photoreceptor_angles_surrounding = photoreceptor_angles_surrounding + rf_offsets

        # Red
        photoreceptor_angles_surrounding_2 = self.chosen_math_library.expand_dims(self.red_photoreceptor_angles, 1)
        photoreceptor_angles_surrounding_2 = self.chosen_math_library.repeat(photoreceptor_angles_surrounding_2, self.n, 1)
        rf_offsets_2 = self.chosen_math_library.linspace(-self.red_photoreceptor_rf_size / 2,
                                                         self.red_photoreceptor_rf_size / 2, num=self.n)
        self.photoreceptor_angles_surrounding_red = photoreceptor_angles_surrounding_2 + rf_offsets_2

        n_photoreceptors_in_computation_axis_0 = self.total_photoreceptor_num

        # Same for both, just requires different dimensions
        mul_for_hypothetical = self.chosen_math_library.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.mul_for_hypothetical = self.chosen_math_library.tile(mul_for_hypothetical,
                                                                  (n_photoreceptors_in_computation_axis_0, self.n, 1, 1))
        add_for_hypothetical = self.chosen_math_library.array(
            [[0, 0], [0, 0], [0, self.width - 1], [self.height - 1, 0]])
        self.add_for_hypothetical = self.chosen_math_library.tile(add_for_hypothetical,
                                                                  (n_photoreceptors_in_computation_axis_0, self.n, 1, 1))

        mul1 = self.chosen_math_library.array([0, 0, 0, 1])
        self.mul1_full = self.chosen_math_library.tile(mul1, (n_photoreceptors_in_computation_axis_0, self.n, 1))

        addition_matrix_unit = self.chosen_math_library.array([0, 0, self.height - 1, self.width - 1])
        self.addition_matrix = self.chosen_math_library.tile(addition_matrix_unit,
                                                             (n_photoreceptors_in_computation_axis_0, self.n, 1))

        conditional_tiled = self.chosen_math_library.array(
            [self.width - 1, self.height - 1, self.width - 1, self.height - 1])
        self.conditional_tiled = self.chosen_math_library.tile(conditional_tiled,
                                                               (n_photoreceptors_in_computation_axis_0, self.n, 1))

        multiplication_matrix_unit = self.chosen_math_library.array([-1, 1, -1, 1])
        self.multiplication_matrix = self.chosen_math_library.tile(multiplication_matrix_unit,
                                                                   (n_photoreceptors_in_computation_axis_0, self.n, 1))

    def update_angles_sigmoid(self, verg_angle, retinal_field, is_left):
        """Set the eyes visual angles to be an even distribution."""

        pr = [0]
        while True:
            spacing = self.sz_rf_spacing + self.density_range / (
                    1 + np.exp(-self.sigmoid_steepness * (pr[-1] - self.sz_size)))
            pr.append(pr[-1] + spacing)
            if pr[-1] > retinal_field:
                break
        pr = pr[:-1]
        pr = self.chosen_math_library.array(pr)
        if is_left:
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
            pr = max_angle - pr
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            pr = pr + min_angle

        return self.chosen_math_library.sort(pr)

    def update_angles(self, verg_angle, retinal_field, is_left, photoreceptor_num, is_uv=False):
        """Set the eyes visual angles to be an even distribution."""
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        return self.chosen_math_library.linspace(min_angle, max_angle, photoreceptor_num)

    def read(self, masked_arena_pixels, eye_x, eye_y, fish_angle, lum_mask, prey_positions, sand_grain_positions,
             proj=True):
        """
        Resolve RF coordinates for each photoreceptor, and use those to sum the relevant pixels.
        """
        # UV Angles with respect to fish (doubled) (PR_N x n)

        photoreceptor_angles_surrounding = self.photoreceptor_angles_surrounding_stacked + fish_angle
        uv_arena_pixels = masked_arena_pixels[:, :, 1:2]
        red_arena_pixels = self.chosen_math_library.concatenate(
            (masked_arena_pixels[:, :, 0:1], masked_arena_pixels[:, :, 2:]), axis=2)
        uv_readings, red_readings = self._read_stacked(masked_arena_pixels_uv=uv_arena_pixels,
                                                       masked_arena_pixels_red=red_arena_pixels,
                                                       eye_x=eye_x,
                                                       eye_y=eye_y,
                                                       photoreceptor_angles_surrounding=photoreceptor_angles_surrounding,
                                                       n_photoreceptors_uv=self.uv_photoreceptor_num,
                                                       n_photoreceptors_red=self.red_photoreceptor_num)

        if len(sand_grain_positions) > 0:
            uv_items = np.concatenate((prey_positions, sand_grain_positions), axis=0)
        else:
            uv_items = prey_positions

        if proj and (len(uv_items)) > 0:
            proj_uv_readings = self._read_prey_proj_parallel(eye_x=eye_x,
                                                             eye_y=eye_y,
                                                             uv_pr_angles=self.uv_photoreceptor_angles,
                                                             fish_angle=fish_angle,
                                                             rf_size=self.uv_photoreceptor_rf_size,
                                                             lum_mask=lum_mask,
                                                             prey_pos=self.chosen_math_library.array(uv_items))

            uv_readings += proj_uv_readings

            if len(sand_grain_positions) > 0:
                red_readings_sand_grains = self._read_prey_proj_parallel(eye_x=eye_x,
                                                                          eye_y=eye_y,
                                                                          uv_pr_angles=self.red_photoreceptor_angles,
                                                                          fish_angle=fish_angle,
                                                                          rf_size=self.red_photoreceptor_rf_size,
                                                                          lum_mask=lum_mask,
                                                                          prey_pos=self.chosen_math_library.array(sand_grain_positions)
                                                                          )

                red_readings += red_readings_sand_grains * self.env_variables["sand_grain_red_component"]

        # uv_readings_scaled = self.scale_readings(uv_readings,
        #                                          self.env_variables['uv_scaling_factor']) * self.env_variables[
        #                                              "light_gain"] / 200)

        self.uv_readings = self.add_noise_to_readings(uv_readings)

        # red_readings_scaled = self.scale_readings(red_readings,
        #                                           self.env_variables['red_scaling_factor'] * self.env_variables[
        #                                               "light_gain"] / 200,
        #                                           self.env_variables['red_2_scaling_factor'] * self.env_variables[
        #                                               "light_gain"] / 200)
        self.red_readings = self.add_noise_to_readings(red_readings)

        # TODO: Bring back after test...
        # interp_uv_readings = self.chosen_math_library.zeros((self.interpolated_observation.shape[0], 1))
        # interp_red_readings = self.chosen_math_library.zeros((self.interpolated_observation.shape[0], 2))
        # interp_uv_readings[:, 0] = self.chosen_math_library.interp(self.interpolated_observation,
        #                                                            self.uv_photoreceptor_angles, self.uv_readings[:, 0])
        # interp_red_readings[:, 0] = self.chosen_math_library.interp(self.interpolated_observation,
        #                                                             self.red_photoreceptor_angles,
        #                                                             self.red_readings[:, 0])
        # interp_red_readings[:, 1] = self.chosen_math_library.interp(self.interpolated_observation,
        #                                                             self.red_photoreceptor_angles,
        #                                                             self.red_readings[:, 1])

        # TODO: Remove after test
        interp_uv_readings = self.chosen_math_library.zeros((self.interpolated_observation.shape[0], 1))
        interp_red_readings = self.chosen_math_library.zeros((self.interpolated_observation.shape[0], 2))
        interp_red_readings[:, 0] = np.array(resize(self.red_readings[:, 0], (100, 1), ))
        interp_uv_readings[:, 0] = np.array(resize(self.uv_readings[:, 0], (100, 1)))
        interp_red_readings[:, 1] = np.array(resize(self.red_readings[:, 1], (100, 1)))

        # Scale for appropriate range
        interp_uv_readings *= self.env_variables["uv_scaling_factor"]
        interp_red_readings[:, 0] *= self.env_variables["red_scaling_factor"]
        interp_red_readings[:, 1] *= self.env_variables["red_2_scaling_factor"]

        self.readings = self.chosen_math_library.concatenate(
            (interp_red_readings[:, 0:1], interp_uv_readings, interp_red_readings[:, 1:]), axis=1)

        if self.using_gpu:
            self.readings = self.readings.get()
        else:
            pass

    def _read_prey_proj(self, eye_x, eye_y, uv_pr_angles, fish_angle, rf_size, lum_mask, prey_pos):
        """Reads the prey projection for the given eye position and fish angle.
        """
        if self.using_gpu:
            abs_uv_pr_angles = uv_pr_angles.get()
        else:
            abs_uv_pr_angles = np.copy(uv_pr_angles)

        rel_prey_pos = prey_pos - np.array([eye_x, eye_y])
        rho = np.hypot(rel_prey_pos[:, 0], rel_prey_pos[:, 1])

        within_range = np.where(rho < self.max_visual_range - 1)[0]
        prey_pos_in_range = prey_pos[within_range, :]
        rel_prey_pos = rel_prey_pos[within_range, :]
        rho = rho[within_range]
        theta = np.arctan2(rel_prey_pos[:, 1], rel_prey_pos[:, 0]) - fish_angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # wrap to [-pi, pi]
        p_num = prey_pos_in_range.shape[0]

        proj = np.zeros((p_num, len(self.ang)))
        for p in range(p_num):
            half_angle = np.arctan(self.prey_diam / (2 * rho[p]))

            l_ind = self._closest_index(self.ang, theta[p] - half_angle)
            r_ind = self._closest_index(self.ang, theta[p] + half_angle)
            try:
                prey_brightness = lum_mask[int(np.floor(prey_pos_in_range[p, 1])) - 1,
                                           int(np.floor(prey_pos_in_range[p, 0])) - 1]  # includes absorption
            except IndexError:
                print(
                    f"Prey Position: {[int(np.floor(prey_pos_in_range[p, 1])) - 1, int(np.floor(prey_pos_in_range[p, 0])) - 1]}")
                print(f"Prey index: {p} of {p_num}")
                print(f"LMS: {lum_mask.shape}")

            if self.using_gpu:
                prey_brightness = prey_brightness.get()

            proj[p, l_ind:r_ind] = prey_brightness

        total_angular_input = np.sum(proj, axis=0)

        pr_input = np.zeros((len(abs_uv_pr_angles), 1))
        for pr in range(len(abs_uv_pr_angles)):
            inds = [self._closest_index(self.ang, abs_uv_pr_angles[pr] - rf_size / 2),
                    self._closest_index(self.ang, abs_uv_pr_angles[pr] + rf_size / 2)]
            pr_input[pr, 0] = np.sum(total_angular_input[inds[0]:inds[1]], axis=0)

        return self.chosen_math_library.array(pr_input)

    def _read_prey_proj_parallel(self, eye_x, eye_y, uv_pr_angles, fish_angle, rf_size, lum_mask, prey_pos):
        """Reads the prey projection for the given eye position and fish angle.
        Same as " but performs more computation in parallel for each prey. Also have removed scatter.
        """

        rel_prey_pos = prey_pos - self.chosen_math_library.array([eye_x, eye_y])
        rho = self.chosen_math_library.hypot(rel_prey_pos[:, 0], rel_prey_pos[:, 1])

        within_range = self.chosen_math_library.where(rho < self.max_visual_range - 1)[0]
        prey_pos_in_range = prey_pos[within_range, :]
        rel_prey_pos = rel_prey_pos[within_range, :]
        rho = rho[within_range]
        theta = self.chosen_math_library.arctan2(rel_prey_pos[:, 1], rel_prey_pos[:, 0]) - fish_angle
        theta = self.chosen_math_library.arctan2(self.chosen_math_library.sin(theta),
                                                 self.chosen_math_library.cos(theta))  # wrap to [-pi, pi]
        p_num = prey_pos_in_range.shape[0]

        half_angle = self.chosen_math_library.arctan(self.prey_diam / (2 * rho))

        l_ind = self._closest_index_parallel(self.ang, theta - half_angle).astype(int)
        r_ind = self._closest_index_parallel(self.ang, theta + half_angle).astype(int)

        prey_brightness = lum_mask[(self.chosen_math_library.floor(prey_pos_in_range[:, 1]) - 1).astype(int),
                                   (self.chosen_math_library.floor(prey_pos_in_range[:, 0]) - 1).astype(
                                       int)]  # includes absorption

        proj = self.chosen_math_library.zeros((p_num, len(self.ang)))

        prey_brightness = self.chosen_math_library.expand_dims(prey_brightness, 1)

        r = self.chosen_math_library.arange(proj.shape[1])
        prey_present = (l_ind[:, None] <= r) & (r_ind[:, None] >= r)
        prey_present = prey_present.astype(float)
        prey_present *= prey_brightness

        total_angular_input = self.chosen_math_library.sum(prey_present, axis=0)

        pr_ind_s = self._closest_index_parallel(self.ang, uv_pr_angles - rf_size / 2)
        pr_ind_e = self._closest_index_parallel(self.ang, uv_pr_angles + rf_size / 2)

        pr_occupation = (pr_ind_s[:, None] <= r) & (pr_ind_e[:, None] >= r)
        pr_occupation = pr_occupation.astype(float)
        pr_input = pr_occupation * self.chosen_math_library.expand_dims(total_angular_input, axis=0)
        pr_input = self.chosen_math_library.sum(pr_input, axis=1)

        return self.chosen_math_library.expand_dims(pr_input, axis=1)

    @staticmethod
    def _closest_index(array, value):
        """Find index of closest value in array."""
        idx = (np.abs(array - value)).argmin()
        return idx

    def _closest_index_parallel(self, array, value_array):
        """Find indices of the closest values in array (for each row in axis=0)."""
        value_array = self.chosen_math_library.expand_dims(value_array, axis=1)
        idxs = (self.chosen_math_library.abs(array - value_array)).argmin(axis=1)
        return idxs

    def _read_stacked(self, masked_arena_pixels_uv, masked_arena_pixels_red, eye_x, eye_y, photoreceptor_angles_surrounding,
                      n_photoreceptors_uv, n_photoreceptors_red):
        """
        Lines method to return pixel sum for all points for each photoreceptor, over its segment.
        Takes photoreceptor_angles_surrounding with UV then Red
        """
        n_photoreceptors = n_photoreceptors_uv + n_photoreceptors_red

        # Make sure angles are in desired range (PR_N x n)
        photoreceptor_angles_surrounding_scaling = (photoreceptor_angles_surrounding // (
                self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        photoreceptor_angles_surrounding = photoreceptor_angles_surrounding + photoreceptor_angles_surrounding_scaling

        # Compute m using tan (PR_N x n)
        m = self.chosen_math_library.tan(photoreceptor_angles_surrounding)

        # Compute c (PR_N x n)
        c = -m * eye_x
        c = c + eye_y

        # Compute components of intersections (PR_N x n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 2)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 2)

        m_mul = self.chosen_math_library.expand_dims(m, 2)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 2)
        m_mul = full_m * self.mul1_full[:n_photoreceptors]
        m_mul[:, :, :3] = 1
        addition_matrix = self.addition_matrix[:n_photoreceptors] * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c_exp * self.multiplication_matrix[
                                            :n_photoreceptors]) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[
                                                               :n_photoreceptors]) + self.add_for_hypothetical[:n_photoreceptors]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:n_photoreceptors]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        try:
            valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates,
                                                                              (n_photoreceptors, self.n, 2, 2))
        except ValueError:
            print("ValueError")
            print(f"Eye position: {[eye_x, eye_y]}")

        # Get intersections (PR_N x 2)
        eye_position = self.chosen_math_library.array([eye_x, eye_y])
        possible_vectors = valid_intersection_coordinates - eye_position

        angles = self.chosen_math_library.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range.
        angle_scaling = (angles // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        angles = angles + angle_scaling

        angles = self.chosen_math_library.round(angles, 3)
        photoreceptor_angles_surrounding = self.chosen_math_library.round(photoreceptor_angles_surrounding, 3)

        photoreceptor_angles_surrounding = self.chosen_math_library.expand_dims(photoreceptor_angles_surrounding, 2)
        photoreceptor_angles_surrounding = self.chosen_math_library.repeat(photoreceptor_angles_surrounding, 2, 2)

        same_values = (angles == photoreceptor_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        try:
            selected_intersections = self.chosen_math_library.reshape(selected_intersections,
                                                                      (n_photoreceptors, self.n, 1, 2))
        except ValueError:
            print(f"Eye position: {eye_position}")

        eye_position_full = self.chosen_math_library.tile(eye_position, (n_photoreceptors, self.n, 1, 1))
        vertices = self.chosen_math_library.concatenate((eye_position_full, selected_intersections), axis=2)
        vertices_xvals = vertices[:, :, :, 0]
        vertices_yvals = vertices[:, :, :, 1]

        min_x = self.chosen_math_library.min(vertices_xvals, axis=2)
        max_x = self.chosen_math_library.max(vertices_xvals, axis=2)
        min_y = self.chosen_math_library.min(vertices_yvals, axis=2)
        max_y = self.chosen_math_library.max(vertices_yvals, axis=2)

        # Threshold with distance from fish
        absolute_min_x = eye_x - self.max_visual_range
        absolute_max_x = eye_x + self.max_visual_range
        absolute_min_y = eye_y - self.max_visual_range
        absolute_max_y = eye_y + self.max_visual_range
        min_x = self.chosen_math_library.clip(min_x, absolute_min_x, self.width)
        max_x = self.chosen_math_library.clip(max_x, 0, absolute_max_x)
        min_y = self.chosen_math_library.clip(min_y, absolute_min_y, self.height)
        max_y = self.chosen_math_library.clip(max_y, 0, absolute_max_y)

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
        full_set = full_set.reshape(n_photoreceptors, -1, 2)

        full_set_uv = full_set[:n_photoreceptors_uv, :]
        full_set_red = full_set[n_photoreceptors_uv:, :]

        # masked_arena_pixels = masked_arena_pixels[
        #     full_set[:, :, 1], full_set[:, :, 0]]  # NOTE: Inverting x and y to match standard in program.
        # total_sum = masked_arena_pixels.sum(axis=1)

        masked_arena_pixels_uv = masked_arena_pixels_uv[full_set_uv[:, :, 1], full_set_uv[:, :,
                                                                              0]]  # NOTE: Inverting x and y to match standard in program. TODO: THIS IS WHERE BREAKS.
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

        total_sum_uv = total_sum_uv * (oversampling_ratio[:n_photoreceptors_uv] / self.n)
        total_sum_red = total_sum_red * (oversampling_ratio_red[n_photoreceptors_uv:] / self.n)

        return total_sum_uv, total_sum_red

    def add_noise_to_readings(self, readings):
        """As specified, adds shot, read, and/or dark noise to readings."""
        if self.env_variables["shot_noise"]:
            photons = self.chosen_math_library.random.poisson(readings)
            # shot_noise_difference = self.chosen_math_library.abs(readings - photons)
            # snr = 1 - self.chosen_math_library.mean(shot_noise_difference / (photons + 1), axis=1)
        else:
            photons = readings

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

        return photons

    def compute_n(self, photoreceptor_rf_size, max_separation=1):
        theta_separation = math.asin(max_separation / self.max_visual_range)
        n = (photoreceptor_rf_size / theta_separation)
        return int(n)

    def get_pr_coverage(self, masked_arena_pixels_uv, masked_arena_pixels_red, eye_x, eye_y, photoreceptor_angles_surrounding,
                        n_photoreceptors_uv, n_photoreceptors_red):
        """For testing purposes"""
        n_photoreceptors = n_photoreceptors_uv + n_photoreceptors_red

        # Make sure angles are in desired range (PR_N x n)
        photoreceptor_angles_surrounding_scaling = (photoreceptor_angles_surrounding // (
                self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        photoreceptor_angles_surrounding = photoreceptor_angles_surrounding + photoreceptor_angles_surrounding_scaling

        # Compute m using tan (PR_N x n)
        m = self.chosen_math_library.tan(photoreceptor_angles_surrounding)

        # Compute c (PR_N x n)
        c = -m * eye_x
        c = c + eye_y

        # Compute components of intersections (PR_N x n x 4)
        c_exp = self.chosen_math_library.expand_dims(c, 2)
        c_exp = self.chosen_math_library.repeat(c_exp, 4, 2)

        m_mul = self.chosen_math_library.expand_dims(m, 2)
        full_m = self.chosen_math_library.repeat(m_mul, 4, 2)
        m_mul = full_m * self.mul1_full[:n_photoreceptors]
        m_mul[:, :, :3] = 1
        addition_matrix = self.addition_matrix[:n_photoreceptors] * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c_exp * self.multiplication_matrix[
                                            :n_photoreceptors]) + addition_matrix) / division_matrix

        intersection_coordinates = self.chosen_math_library.expand_dims(intersection_components, 3)
        intersection_coordinates = self.chosen_math_library.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * self.mul_for_hypothetical[
                                                               :n_photoreceptors]) + self.add_for_hypothetical[:n_photoreceptors]

        # Compute possible intersections (PR_N x 2 x 2 x 2)
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < self.conditional_tiled[:n_photoreceptors]) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = self.chosen_math_library.reshape(valid_intersection_coordinates,
                                                                          (n_photoreceptors, self.n, 2, 2))
        # Get intersections (PR_N x 2)
        eye_position = self.chosen_math_library.array([eye_x, eye_y])
        possible_vectors = valid_intersection_coordinates - eye_position

        angles = self.chosen_math_library.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range.
        angle_scaling = (angles // (self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        angles = angles + angle_scaling

        angles = self.chosen_math_library.round(angles, 3)
        photoreceptor_angles_surrounding = self.chosen_math_library.round(photoreceptor_angles_surrounding, 3)

        photoreceptor_angles_surrounding = self.chosen_math_library.expand_dims(photoreceptor_angles_surrounding, 2)
        photoreceptor_angles_surrounding = self.chosen_math_library.repeat(photoreceptor_angles_surrounding, 2, 2)

        same_values = (angles == photoreceptor_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = self.chosen_math_library.reshape(selected_intersections, (n_photoreceptors, self.n, 1, 2))

        eye_position_full = self.chosen_math_library.tile(eye_position, (n_photoreceptors, self.n, 1, 1))
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
        full_set = full_set.reshape(n_photoreceptors, -1, 2)

        full_set_uv = full_set[:n_photoreceptors_uv, :]
        full_set_red = full_set[n_photoreceptors_uv:, :]

        return full_set_uv, full_set_red

    def get_pr_line_coordinates_uv(self, eye_x, eye_y):
        """For testing purposes"""
        # Make sure angles are in desired range (PR_N x n)
        photoreceptor_angles_surrounding_scaling = (self.uv_photoreceptor_angles // (
                self.chosen_math_library.pi * 2)) * self.chosen_math_library.pi * -2
        photoreceptor_angles_surrounding = self.uv_photoreceptor_angles + photoreceptor_angles_surrounding_scaling

        # Compute m using tan (PR_N x n)
        m = self.chosen_math_library.tan(photoreceptor_angles_surrounding)

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
                                                               :self.uv_photoreceptor_num,
                                                               0]) + self.add_for_hypothetical[
                                                                     :self.uv_photoreceptor_num, 0]

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
        photoreceptor_angles_surrounding = self.chosen_math_library.round(photoreceptor_angles_surrounding, 3)

        photoreceptor_angles_surrounding = self.chosen_math_library.expand_dims(photoreceptor_angles_surrounding, 1)
        photoreceptor_angles_surrounding = self.chosen_math_library.repeat(photoreceptor_angles_surrounding, 2, 1)

        same_values = (angles == photoreceptor_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = self.chosen_math_library.reshape(selected_intersections,
                                                                  (self.uv_photoreceptor_num, 1, 2))

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
