import numpy as np
import math
import cupy as cp

from matplotlib import path

from Tools.Sectors.sector_sum import sector_sum
from Tools.Lines.lines import lines


class NewVisFan:

    def __init__(self, board, verg_angle, retinal_field, is_left, num_arms, min_distance, max_distance, dark_gain,
                 light_gain, bkg_scatter, dark_col):
        self.num_arms = num_arms
        self.distances = np.array([min_distance, max_distance])

        self.vis_angles = None
        self.dist = None
        self.theta = None

        self.update_angles(verg_angle, retinal_field, is_left)
        self.readings = np.zeros((num_arms, 3), 'int')
        self.board = board
        self.dark_gain = dark_gain
        self.light_gain = light_gain
        self.bkg_scatter = bkg_scatter
        self.dark_col = dark_col

        self.width, self.height = self.board.get_size()

        # TODO: Make parameters:
        self.photoreceptor_num = num_arms
        self.photoreceptor_rf_size = 0.014
        self.retinal_field_size = retinal_field
        self.photoreceptor_spacing = self.retinal_field_size/self.photoreceptor_num

        # Create matrix A of coordinates (w.h.3.2). Allows checking whether points are in triangles.
        # xp, yp = np.meshgrid(range(self.width), range(self.height))
        # xp = np.expand_dims(xp, 2)
        # yp = np.expand_dims(yp, 2)
        # xp = np.expand_dims(xp, 3)
        # yp = np.expand_dims(yp, 3)
        # coordinates = np.concatenate((xp, yp), 3)
        # self.rearranged_coordinates = np.repeat(coordinates, repeats=3, axis=2)

        # For checking points in sector when is quadrilateral. TODO: Find way of doing more efficiently
        # self.all_points = [[x, y] for x in range(1500) for y in range(1500)]

    def cartesian(self, bx, by, bangle):
        x = bx + self.dist * np.cos(self.theta + bangle)
        y = (by + self.dist * np.sin(self.theta + bangle))
        return x, y

    def update_angles(self, verg_angle, retinal_field, is_left):
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        self.vis_angles = np.linspace(min_angle, max_angle, self.num_arms)
        # self.dist, self.theta = np.meshgrid(self.distances, self.vis_angles)

    def segment_method(self, masked_arena_pixels, fish_x, fish_y, fish_angle):
        all_vertices = self.compute_all_sectors(fish_x, fish_y, fish_angle)
        top_left, bottom_left, top_right, bottom_right = self.get_corner_sectors(all_vertices)
        for i, channel_angle in enumerate(self.vis_angles):
            vertices = all_vertices[i, :, :]
            vertices = self.get_extra_vertices(i, vertices, top_left, bottom_left, top_right, bottom_right)

            if len(vertices) == 3:
                vertices = sorted(vertices, key=lambda x: x[0])
                segment_sum = sector_sum(self.rearranged_coordinates, np.array(vertices), masked_arena_pixels)

                # segment_sum = self.sum_within_triangle(masked_arena_pixels, vertices)
            else:
                t1 = [vertices[0]] + [vertices[2]] + [vertices[1]]
                t2 = [vertices[0]] + vertices[2:]
                # TODO: Fix problem by including lines - will always ignore points on the line.
                segment_sum1 = self.sum_within_triangle(masked_arena_pixels, t1)
                segment_sum2 = self.sum_within_triangle(masked_arena_pixels, t2)
                # segment_sum = self.sum_within_polygon(masked_arena_pixels, vertices)
                segment_sum = segment_sum1 + segment_sum2
            print(segment_sum)
            self.readings[i] = segment_sum #* 100000  # TODO: remove scaling once calibrated visual system.

    def read(self, masked_arena_pixels, fish_x, fish_y, fish_angle):
        # Lines version:
        # self.lines_method(masked_arena_pixels, fish_x, fish_y, fish_angle)
        self.lines_method_cupy(cp.array(masked_arena_pixels), fish_x, fish_y, fish_angle)

        # Old version
        # self.segment_method(masked_arena_pixels, fish_x, fish_y, fish_angle)

    def lines_method_cupy(self, masked_arena_pixels, fish_x, fish_y, fish_angle, n=20):
        # Angles with respect to fish (doubled) (120 x n)
        channel_angles = cp.array(self.vis_angles) + fish_angle
        channel_angles_surrounding = cp.expand_dims(channel_angles, 1)
        channel_angles_surrounding = cp.repeat(channel_angles_surrounding, n, 1)  # TODO: Can reducec to use omly once then add fish angle ach time. (also combine with below)

        # Angles of each side (120 x n)
        rf_offsets = cp.linspace(-self.photoreceptor_rf_size/2, self.photoreceptor_rf_size/2, num=n)  # TODO: Use only once.
        channel_angles_surrounding = channel_angles_surrounding + rf_offsets

        # Make sure is in desired range (120 x n) TODO: might need to find way of doing it multiple times e.g. by // operation
        below_range = (channel_angles_surrounding < 0) * cp.pi * 2
        channel_angles_surrounding = channel_angles_surrounding + below_range
        above_range = (channel_angles_surrounding > cp.pi * 2) * -cp.pi*2
        channel_angles_surrounding = channel_angles_surrounding + above_range

        # Compute m using tan (120 x n)
        m = cp.tan(channel_angles_surrounding)

        # Compute c (120 x n)
        c = -m * fish_x
        c = c + fish_y

        # Compute components of intersections (120 x n x 4)
        c_exp = cp.expand_dims(c, 2)
        c_exp = cp.repeat(c_exp, 4, 2)

        multiplication_matrix_unit = cp.array([-1, 1, -1, 1])
        multiplication_matrix = cp.tile(multiplication_matrix_unit, (120, n, 1))

        addition_matrix_unit = cp.array([0, 0, self.height-1, self.width-1])
        addition_matrix = cp.tile(addition_matrix_unit, (120, n, 1))

        mul1 = cp.array([0, 0, 0, 1])
        mul1_full = cp.tile(mul1, (120, n, 1))
        m_mul = cp.expand_dims(m, 2)
        full_m = cp.repeat(m_mul, 4, 2)
        m_mul = full_m * mul1_full
        m_mul[:, :, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c_exp * multiplication_matrix) + addition_matrix)/division_matrix

        mul_for_hypothetical = cp.array([[1, 0], [0, 1], [1, 0], [0, 1]])  # TODO: Do only once
        mul_for_hypothetical = cp.tile(mul_for_hypothetical, (120, n, 1, 1))  # TODO: Do only once
        add_for_hypothetical = cp.array([[0, 0], [0, 0], [0, self.width-1], [self.height-1, 0]]) # TODO: Do only once
        add_for_hypothetical = cp.tile(add_for_hypothetical, (120, n, 1, 1))  # TODO: Do only once

        intersection_coordinates = cp.expand_dims(intersection_components, 3)
        intersection_coordinates = cp.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (120 x 2 x 2 x 2)
        conditional_tiled = cp.array([self.width-1, self.height-1, self.width-1, self.height-1])  # TODO: Do only once
        conditional_tiled = cp.tile(conditional_tiled, (120, n, 1))  # TODO: Do only once
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = cp.reshape(valid_intersection_coordinates, (120, n, 2, 2))

        # Get intersections (120 x 2)
        eye_position = cp.array([fish_x, fish_y])  # TODO: Do only once
        possible_vectors = valid_intersection_coordinates - eye_position
        angles = cp.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        below_range = (angles < 0) * cp.pi * 2
        angles = angles + below_range
        above_range = (angles > cp.pi * 2) * -cp.pi*2
        angles = angles + above_range

        angles = cp.round(angles, 2)
        channel_angles_surrounding = cp.round(channel_angles_surrounding, 2)

        channel_angles_surrounding = cp.expand_dims(channel_angles_surrounding, 2)
        channel_angles_surrounding = cp.repeat(channel_angles_surrounding, 2, 2)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = cp.reshape(selected_intersections, (120, n, 1, 2))

        fish_position_full = cp.tile(cp.array([fish_x, fish_y]), (120, n, 1, 1))
        vertices = cp.concatenate((fish_position_full, selected_intersections), axis=2)
        vertices_xvals = vertices[:, :, :, 0]
        vertices_yvals = vertices[:, :, :, 1]

        # TODO: Probably faster way of doing below...
        min_x = cp.min(vertices_xvals, axis=2)
        max_x = cp.max(vertices_xvals, axis=2)
        min_y = cp.min(vertices_yvals, axis=2)
        max_y = cp.max(vertices_yvals, axis=2)

        # self.readings = lines(masked_arena_pixels, m, c, min_x, max_x, min_y, max_y)
        # self.compute_segment_sums_line(masked_arena_pixels, m, c, min_x, max_x, min_y, max_y)
        self.compute_segment_sums_line_cupy(masked_arena_pixels, m, c, min_x, max_x, min_y, max_y)

    def lines_method(self, masked_arena_pixels, fish_x, fish_y, fish_angle, n=20):
        # Angles with respect to fish (doubled) (120 x n)
        channel_angles = self.vis_angles + fish_angle
        channel_angles_surrounding = np.expand_dims(channel_angles, 1)
        channel_angles_surrounding = np.repeat(channel_angles_surrounding, n, 1)

        # Angles of each side (120 x n)
        rf_offsets = np.linspace(-self.photoreceptor_rf_size/2, self.photoreceptor_rf_size/2, num=n)
        channel_angles_surrounding = channel_angles_surrounding + rf_offsets

        # Make sure is in desired range (120 x n) TODO: might need to find way of doing it multiple times e.g. by // operation
        below_range = (channel_angles_surrounding < 0) * np.pi * 2
        channel_angles_surrounding = channel_angles_surrounding + below_range
        above_range = (channel_angles_surrounding > np.pi * 2) * -np.pi*2
        channel_angles_surrounding = channel_angles_surrounding + above_range

        # Compute m using tan (120 x n)
        m = np.tan(channel_angles_surrounding)

        # Compute c (120 x n)
        c = -m * fish_x
        c = c + fish_y

        # Compute components of intersections (120 x n x 4)
        c_exp = np.expand_dims(c, 2)
        c_exp = np.repeat(c_exp, 4, 2)


        multiplication_matrix_unit = np.array([-1, 1, -1, 1])
        multiplication_matrix = np.tile(multiplication_matrix_unit, (120, n, 1))

        addition_matrix_unit = np.array([0, 0, self.height-1, self.width-1])
        addition_matrix = np.tile(addition_matrix_unit, (120, n, 1))

        mul1 = np.array([0, 0, 0, 1])
        mul1_full = np.tile(mul1, (120, n, 1))
        m_mul = np.expand_dims(m, 2)
        full_m = np.repeat(m_mul, 4, 2)
        m_mul = full_m * mul1_full
        m_mul[:, :, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, :, 1] = 1
        division_matrix[:, :, 3] = 1

        intersection_components = ((c_exp * multiplication_matrix) + addition_matrix)/division_matrix

        mul_for_hypothetical = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        mul_for_hypothetical = np.tile(mul_for_hypothetical, (120, n, 1, 1))
        add_for_hypothetical = np.array([[0, 0], [0, 0], [0, self.width-1], [self.height-1, 0]])
        add_for_hypothetical = np.tile(add_for_hypothetical, (120, n, 1, 1))

        intersection_coordinates = np.expand_dims(intersection_components, 3)
        intersection_coordinates = np.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (120 x 2 x 2 x 2)
        conditional_tiled = np.array([self.width-1, self.height-1, self.width-1, self.height-1])
        conditional_tiled = np.tile(conditional_tiled, (120, n, 1))
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = np.reshape(valid_intersection_coordinates, (120, n, 2, 2))
        # try:
        #     valid_intersection_coordinates = np.reshape(valid_intersection_coordinates, (120, n, 2, 2))
        # except ValueError:
        #     x = True

        # Get intersections (120 x 2)
        eye_position = np.array([fish_x, fish_y])
        possible_vectors = valid_intersection_coordinates - eye_position
        angles = np.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        below_range = (angles < 0) * np.pi * 2
        angles = angles + below_range
        above_range = (angles > np.pi * 2) * -np.pi*2
        angles = angles + above_range

        angles = np.round(angles, 2)
        channel_angles_surrounding = np.round(channel_angles_surrounding, 2)

        channel_angles_surrounding = np.expand_dims(channel_angles_surrounding, 2)
        channel_angles_surrounding = np.repeat(channel_angles_surrounding, 2, 2)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = np.reshape(selected_intersections, (120, n, 1, 2))

        fish_position_full = np.tile([fish_x, fish_y], (120, n, 1, 1))
        vertices = np.concatenate((fish_position_full, selected_intersections), axis=2)
        vertices_xvals = vertices[:, :, :, 0]
        vertices_yvals = vertices[:, :, :, 1]

        min_x = np.min(vertices_xvals, axis=2)
        max_x = np.max(vertices_xvals, axis=2)
        min_y = np.min(vertices_yvals, axis=2)
        max_y = np.max(vertices_yvals, axis=2)

        # self.readings = lines(masked_arena_pixels, m, c, min_x, max_x, min_y, max_y)
        # self.compute_segment_sums_line(masked_arena_pixels, m, c, min_x, max_x, min_y, max_y)
        self.compute_segment_sums_line(masked_arena_pixels, m, c, min_x, max_x, min_y, max_y)

    def compute_all_sectors(self, fish_x, fish_y, fish_angle):
        # Angles with respect to fish (doubled) (120 x 2)
        channel_angles = self.vis_angles + fish_angle
        channel_angles_surrounding = np.expand_dims(channel_angles, 1)
        channel_angles_surrounding = np.repeat(channel_angles_surrounding, 2, 1)

        # Angles of each side (120 x 2)
        rf_offsets = np.array([-self.photoreceptor_rf_size/2, self.photoreceptor_rf_size/2])  # TODO: can move to init.
        channel_angles_surrounding = channel_angles_surrounding + rf_offsets

        # Make sure is in desired range (120 x 2) TODO: might need to find way of doing it multiple times e.g. by // operation
        below_range = (channel_angles_surrounding < 0) * np.pi * 2
        channel_angles_surrounding = channel_angles_surrounding + below_range
        above_range = (channel_angles_surrounding > np.pi * 2) * -np.pi*2
        channel_angles_surrounding = channel_angles_surrounding + above_range

        # Compute m using tan (120 x 2)
        m = np.tan(channel_angles_surrounding)

        # Compute c (120 x 2)
        c = -m * fish_x
        c = c + fish_y

        # Compute components of intersections (120 x 2 x 4)
        c = np.expand_dims(c, 2)
        c = np.repeat(c, 4, 2)

        multiplication_matrix_unit = np.array([-1, 1, -1, 1])
        multiplication_matrix = np.tile(multiplication_matrix_unit, (120, 2, 1))

        addition_matrix_unit = np.array([0, 0, self.height, self.width])
        addition_matrix = np.tile(addition_matrix_unit, (120, 2, 1))

        mul1 = np.array([0, 0, 0, 1])
        mul1_full = np.tile(mul1, (120, 2, 1))
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
        mul_for_hypothetical = np.tile(mul_for_hypothetical, (120, 2, 1, 1))
        add_for_hypothetical = np.array([[0, 0], [0, 0], [0, self.width], [self.height, 0]])
        add_for_hypothetical = np.tile(add_for_hypothetical, (120, 2, 1, 1))

        intersection_coordinates = np.expand_dims(intersection_components, 3)
        intersection_coordinates = np.repeat(intersection_coordinates, 2, 3)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (120 x 2 x 2 x 2)
        conditional_tiled = np.array([self.width, self.height, self.width, self.height])
        conditional_tiled = np.tile(conditional_tiled, (120, 2, 1))
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]
        valid_intersection_coordinates = np.reshape(valid_intersection_coordinates, (120, 2, 2, 2))

        # Get intersections (120 x 2)
        eye_position = np.array([fish_x, fish_y])
        possible_vectors = valid_intersection_coordinates - eye_position
        angles = np.arctan2(possible_vectors[:, :, :, 1], possible_vectors[:, :, :, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        below_range = (angles < 0) * np.pi * 2
        angles = angles + below_range
        above_range = (angles > np.pi * 2) * -np.pi*2
        angles = angles + above_range

        angles = np.round(angles, 2)
        channel_angles_surrounding = np.round(channel_angles_surrounding, 2)

        channel_angles_surrounding = np.expand_dims(channel_angles_surrounding, 2)
        channel_angles_surrounding = np.repeat(channel_angles_surrounding, 2, 2)

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]
        selected_intersections = np.reshape(selected_intersections, (120, 2, 2))

        fish_position_full = np.tile([fish_x, fish_y], (120, 1, 1))
        vertices = np.concatenate((fish_position_full, selected_intersections), axis=1)

        return vertices

    def compute_channel_vertices(self, eye_position, fish_angle, channel_orientation, channel_rf_size):
        """Given position of an eye, angle of fish, orientation of a channel with respect to the fish, and the angular
        size of that channels receptive field, returns the vertices of the shape (triangle or quadrilateral), enclosed
        by walls, formed by that channels receptive field."""

        # Compute absolute angles enclosing channel receptive field
        angle_a, angle_b = self.get_channel_rf_angles(fish_angle, channel_orientation, channel_rf_size)

        # Compute parameters of lines from angles of fish.
        m = np.tan([angle_a, angle_b])
        c_a = eye_position[1] - m[0] * eye_position[0]
        c_b = eye_position[1] - m[1] * eye_position[0]

        # Find all intersections of lines with walls
        intersections_a = self.find_wall_intersections(m[0], c_a)
        intersections_b = self.find_wall_intersections(m[1], c_b)

        # Gets the desired intersections from each (so that in direction fish is facing)
        intersections = [eye_position]
        intersections += self.get_valid_intersections(intersections_a, eye_position, angle_a)
        intersections += self.get_valid_intersections(intersections_b, eye_position, angle_b)

        intersections = self.check_if_needs_corner(intersections)  # Old, have new method

        return intersections

    @staticmethod
    def get_channel_rf_angles(fish_angle, channel_orientation, channel_rf_size):
        """Flips direction of angles on x axis (for gradient calculations), then ensures all values are represented in
        positive way."""
        # Get absolute angle of theta in old angle system.
        absolute_channel_theta = fish_angle + channel_orientation

        # DONT NEED Swap angle to new orientation, to allow computation of gradient. Involves reflecting angle on x axis.
        # absolute_channel_theta = (np.pi * 2) - absolute_channel_theta

        angle_a = absolute_channel_theta - (channel_rf_size/2)
        angle_b = absolute_channel_theta + (channel_rf_size/2)

        # Ensure all angles are between 0 and 2pi
        while angle_a < 0:
            angle_a += 2 * np.pi
        while angle_b < 0:
            angle_b += 2 * np.pi

        while angle_a > 2 * np.pi:
            angle_a -= 2 * np.pi
        while angle_b > 2 * np.pi:
            angle_b -= 2 * np.pi

        return angle_a, angle_b

    def find_wall_intersections(self, m, c):
        w = self.board.width
        h = self.board.height
        intersections = []

        # At y=0
        x0 = -c/m
        if 0 <= x0 <= w:
            intersections.append([x0, 0])

        # At x=0
        y0 = c
        if 0 <= y0 <= h:
            intersections.append([0, y0])

        # At y=h
        x1 = (h - c)/m
        if 0 <= x1 <= w:
            intersections.append([x1, h])

        # At x=w
        y1 = w * m + c
        if 0 <= y1 <= h:
            intersections.append([w, y1])

        return intersections

    @staticmethod
    def get_valid_intersections(intersections, fish_position, angle):
        vector_1 = [intersections[0][0]-fish_position[0], intersections[0][1]-fish_position[1]]
        vector_2 = [intersections[1][0]-fish_position[0], intersections[1][1]-fish_position[1]]
        angle_1 = math.atan2(vector_1[1], vector_1[0])
        angle_2 = math.atan2(vector_2[1], vector_2[0])

        while angle_1 < 0:
            angle_1 += 2 * np.pi
        while angle_2 < 0:
            angle_2 += 2 * np.pi

        if round(angle_1, 2) == round(angle, 2):
            return [intersections[0]]
        elif round(angle_2, 2) == round(angle, 2):
            return [intersections[1]]
        else:
            print(f"Angle 1: {angle_1}, angle 2: {angle_2}, angle: {angle}")
            print("ERROR, INVALID ANGLE CALCULATION")

    def get_corner_sectors(self, vertices):
        x_vertices = vertices[:, :, 0]
        y_vertices = vertices[:, :, 1]

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
        return top_left, bottom_left, top_right, bottom_right

    def get_extra_vertices(self, i, vertices, top_left, bottom_left, top_right, bottom_right):
        vertices = vertices.tolist()
        if top_left[i]:
            vertices.insert(2, [0, 0])
        elif bottom_left[i]:
            vertices.insert(2, [0, self.height])

            # vertices = vertices + [[0, self.height]]
        elif top_right[i]:
            vertices.insert(2, [self.width, 0])
            # vertices = vertices + [[self.width, 0]]
        elif bottom_right[i]:
            vertices.insert(2, [self.width, self.height])

            # vertices = vertices + [[self.width, self.height]]
        return vertices

    @staticmethod
    def check_cross_product(ab, bc, ca, ap, bp, cp):
        """No longer used. Computes third component of cross products and returns true if they have the same sign"""
        a_cp = ab[0]*ap[1] - ab[1]*ap[0]
        b_cp = bc[0]*bp[1] - bc[1]*bp[0]
        c_cp = ca[0]*cp[1] - ca[1]*cp[0]
        return (a_cp < 0) == (b_cp < 0) == (c_cp < 0)

    def sum_within_triangle_mpl(self, masked_arena_pixels, vertices):
        """Much slower than numpy version."""
        p = path.Path(vertices)
        b = p.contains_points(self.all_points)
        sector_points = np.reshape(b, (1500, 1500)) * 1
        sector_points = np.expand_dims(sector_points, 2)
        sector_points = np.repeat(sector_points, 3, 2)
        weighted_points = sector_points * masked_arena_pixels

        # Sum values from entire matrix along all but final axis (3)
        total_sum = weighted_points.sum(axis=(0, 1))
        print(total_sum)
        return total_sum

    def sum_within_triangle(self, masked_arena_pixels, vertices):
        # Sort vertices in ascending order of x value.
        vertices = sorted(vertices, key=lambda x: x[0])

        # Define triangle ABC, which have ascending x values for vertices. TODO: unpack more efficiently
        xa = vertices[0][0]
        xb = vertices[1][0]
        xc = vertices[2][0]
        ya = vertices[0][1]
        yb = vertices[1][1]
        yc = vertices[2][1]

        # Create vectors for triangle sides
        ab = [xb-xa, yb-ya]
        bc = [xc-xb, yc-yb]
        ca = [xa-xc, ya-yc]

        # Create matrix B of triangle vertices (w.h.3.2)
        repeating_unit = np.array([[xa, ya], [xb, yb], [xc, yc]])
        # full_field = np.expand_dims(repeating_unit, axis=0)
        # full_field = np.expand_dims(full_field, axis=0)
        # full_field = np.repeat(full_field, self.width, 0)
        # full_field = np.repeat(full_field, self.height, 1)
        # full_field = np.tile(repeating_unit, (1500, 1500, 1, 1))
        # TODO: check that coordinates are correct in case of differing width and height

        xmin = round(min(np.array(vertices)[:, 0]))
        xmax = round(max(np.array(vertices)[:, 0]))
        ymin = round(min(np.array(vertices)[:, 1]))
        ymax = round(max(np.array(vertices)[:, 1]))
        coordinates_to_test = self.rearranged_coordinates[ymin:ymax, xmin:xmax, :]
        full_field = np.tile(repeating_unit, (coordinates_to_test.shape[0], coordinates_to_test.shape[1], 1, 1))

        # Compute C = A-B (corresponds to subtracting vertices points from each coordinate in space (w.h.3.2)
        new_vector_points = coordinates_to_test - full_field

        # Flip C along final axis (so y is where x was previously) (w.h.3.2)
        new_vector_points_flipped = np.flip(new_vector_points, 3)

        # Create matrix D with repeated triangle side vectors (w.h.3.2)
        old_vector_points = [ab, bc, ca]
        # old_vector_points = np.expand_dims(old_vector_points, axis=0)
        # old_vector_points = np.expand_dims(old_vector_points, axis=0)
        # old_vector_points = np.repeat(old_vector_points, 1500, 0)
        # old_vector_points = np.repeat(old_vector_points, 1500, 1)
        # old_vector_points = np.tile(old_vector_points, (1500, 1500, 1, 1))
        old_vector_points = np.tile(old_vector_points, (coordinates_to_test.shape[0], coordinates_to_test.shape[1], 1, 1))

        # Perform E = C * D (w.h.3.2)
        cross_product_components = new_vector_points_flipped * old_vector_points

        # Subtract y from x values (final axis) from E (w.h.3)
        cross_product = cross_product_components[:, :, :, 0] - cross_product_components[:, :, :, 1]

        # Artificial tally for debugging.
        # artificial_tally = []
        # for i in range(len(cross_product)):
        #     for j in range(len(cross_product[i, :, :])):
        #         if (cross_product[i, j, 0] < 0) == (cross_product[i, j, 1] < 0) == (cross_product[i, j, 2] < 0):
        #             artificial_tally.append([i, j])

        # If points in cross product are negative, set equal to 1, else 0. (w.h.3)
        cross_product_less_than = (cross_product < 0) * 1

        # Along 3rd axis, sum values.
        cross_product_boolean_axis = np.sum(cross_product_less_than, axis=2)

        # Set points to 1 if that point cross product sum is 0 or 3.
        cross_product_boolean_axis_sum = ((cross_product_boolean_axis == 0) | (cross_product_boolean_axis == 3)) * 1

        # Expand enclosed points to dimensions in order to multiply by mask. (w.h.3)
        sector_points = np.expand_dims(cross_product_boolean_axis_sum, 2)
        sector_points = np.repeat(sector_points, 3, 2)

        # Multiply enclosion mask by pixel mask (w.h.3)
        masked_arena_pixels = masked_arena_pixels[ymin:ymax, xmin:xmax, :]
        weighted_points = sector_points * masked_arena_pixels

        # Sum values from entire matrix along all but final axis (3)
        total_sum = weighted_points.sum(axis=(0, 1))

        return total_sum

    def sum_within_triangle_old(self, masked_arena_pixels, vertices):
        """Old method, much slower. Kept for debugging purposes"""
        # Define triangle ABC, which have ascending x values for vertices.
        xa = vertices[0][0]
        xb = vertices[1][0]
        xc = vertices[2][0]
        ya = vertices[0][1]
        yb = vertices[1][1]
        yc = vertices[2][1]

        # Create vectors for triangle sides
        ab = [xb-xa, yb-ya]
        bc = [xc-xb, yc-yb]
        ca = [xa-xc, ya-yc]

        all_points_1 = [[x, y] for x in range(1500) for y in range(1500)]
        points_in_triangle = []
        for point in all_points_1:
            ap = [point[0]-xa, point[1]-ya]
            bp = [point[0]-xb, point[1]-yb]
            cp = [point[0]-xc, point[1]-yc]

            if self.check_cross_product(ab, bc, ca, ap, bp, cp):
                points_in_triangle.append(point)
        sum = [0, 0, 0]
        for point in points_in_triangle:
            sum += masked_arena_pixels[point[0], point[1], :]
        return sum

    def sum_within_polygon(self, masked_arena_pixels, vertices):
        p = path.Path(vertices)
        b = p.contains_points(self.all_points)
        sector_points = np.reshape(b, (1500, 1500)) * 1
        sector_points = np.expand_dims(sector_points, 2)
        sector_points = np.repeat(sector_points, 3, 2)
        weighted_points = sector_points * masked_arena_pixels

        # Sum values from entire matrix along all but final axis (3)
        total_sum = weighted_points.sum(axis=(0, 1))
        return total_sum

    @staticmethod
    def remove_green_channel(board):
        """Removes the green channel, which is not used."""
        board = np.delete(board, 1, 2)
        return board

    def get_all_sectors(self, fish_position, fish_orientation):
        """Returns list of all the sectors in that part of visual field."""
        new_sectors = self.compute_all_sectors(fish_position[0], fish_position[1], fish_orientation)

        return new_sectors

    def compute_all_segment_sums(self, all_sectors, masked_arena_pixels):
        """Not used, causes crash."""
        # Sort vertices in ascending order of x value.
        # vertices = sorted(all_sectors, key=lambda x: x[0][0])
        all_sectors_x = all_sectors[:, :, 0]
        sorted_order = np.argsort(all_sectors_x, axis=1)
        sorted_order = np.expand_dims(sorted_order, 2)
        sorted_order = np.repeat(sorted_order, 2, 2)
        static_indices = np.indices((120, 3, 2))

        vertices = all_sectors[static_indices[0], sorted_order, static_indices[2]]

        # Define triangle ABC, which have ascending x values for vertices.
        xa = vertices[:, 0, 0]
        xb = vertices[:, 1, 0]
        xc = vertices[:, 2, 0]
        ya = vertices[:, 0, 1]
        yb = vertices[:, 1, 1]
        yc = vertices[:, 2, 1]

        # Create vectors for triangle sides
        ab = [xb - xa, yb - ya]
        bc = [xc - xb, yc - yb]
        ca = [xa - xc, ya - yc]

        # Create matrix B of triangle vertices (w.h.3.2)
        repeating_unit = np.array([[xa, ya], [xb, yb], [xc, yc]])
        repeating_unit = np.rollaxis(repeating_unit, 2, 0)
        full_field = np.tile(repeating_unit, (1500, 1500, 1, 1, 1))
        # TODO: check that coordinates are correct in case of differing width and height

        # Compute C = A-B (corresponds to subtracting vertices points from each coordinate in space (w.h.3.2)
        new_vector_points = self.rearranged_coordinates - full_field

        # Flip C along final axis (so y is where x was previously) (w.h.3.2)
        new_vector_points_flipped = np.flip(new_vector_points, 3)

        # Create matrix D with repeated triangle side vectors (w.h.3.2)
        old_vector_points = [ab, bc, ca]
        old_vector_points = np.tile(old_vector_points, (1500, 1500, 1, 1))

        # Perform E = C * D (w.h.3.2)
        cross_product_components = new_vector_points_flipped * old_vector_points

        # Subtract y from x values (final axis) from E (w.h.3)
        cross_product = cross_product_components[:, :, :, 0] - cross_product_components[:, :, :, 1]

        # If points in cross product are negative, set equal to 1, else 0. (w.h.3)
        cross_product_less_than = (cross_product < 0) * 1

        # Along 3rd axis, sum values.
        cross_product_boolean_axis = np.sum(cross_product_less_than, axis=2)

        # Set points to 1 if that point cross product sum is 0 or 3.
        cross_product_boolean_axis_sum = ((cross_product_boolean_axis == 0) | (cross_product_boolean_axis == 3)) * 1

        # Expand enclosed points to dimensions in order to multiply by mask. (w.h.3)
        sector_points = np.expand_dims(cross_product_boolean_axis_sum, 2)
        sector_points = np.repeat(sector_points, 3, 2)

        # Multiply enclosion mask by pixel mask (w.h.3)
        weighted_points = sector_points * masked_arena_pixels

        # Sum values from entire matrix along all but final axis (3)
        total_sum = weighted_points.sum(axis=(0, 1))

        return total_sum

    @staticmethod
    def get_points_along_line(m, c, xmin, xmax, ymin, ymax):
        """Returns all integer points that are touched by line. Looks on both y and x ranges to avoid precision problems."""
        # Precision is problem.
        xmin = np.floor(xmin)
        xmax = np.floor(xmax)
        ymin = np.floor(ymin)
        ymax = np.floor(ymax)
        xrange = np.arange(xmin, xmax+1)
        yvals = (m * xrange) + c
        yvals = np.around(yvals)
        set1 = np.stack((xrange, yvals), axis=-1)

        yrange = np.arange(ymin, ymax+1)
        xvals = (yrange - c) / m
        xvals = np.around(xvals)
        set2 = np.stack((xvals, yrange), axis=-1)

        full_set = np.vstack((set1, set2))
        full_set = np.unique(full_set, axis=0)

        return full_set

    def compute_segment_sums_line(self, masked_arena_pixels, m, c, xmin, xmax, ymin, ymax):
        # c = c[:, :, 0]
        # xmin = np.floor(xmin)
        # xmax = np.floor(xmax)
        # ymin = np.floor(ymin)
        # ymax = np.floor(ymax)

        n_photoreceptors = m.shape[0]

        # Fully vectorised (3x slower than iterative).
        # x_len = np.max(np.rint(xmax[:, 0] - xmin[:, 0]).astype(int))
        # y_len = np.max(np.rint(ymax[:, 0] - ymin[:, 0]).astype(int))
        #
        # x_ranges = np.linspace(xmin, xmax, x_len)
        # y_ranges = np.linspace(ymin, ymax, y_len)
        #
        # y_values = (m * x_ranges) + c
        # y_values = np.floor(y_values)
        # set_1 = np.stack((x_ranges, y_values), axis=-1)
        # x_values = (y_ranges - c) / m
        # x_values = np.floor(x_values)
        # set_2 = np.stack((x_values, y_ranges), axis=-1)
        # full_set = np.vstack((set_1, set_2))
        # full_set = full_set.reshape(n_photoreceptors, -1, full_set.shape[-1]).astype(int)
        #
        # used_pixels = masked_arena_pixels[full_set[:, :, 0], full_set[:, :, 1]]
        # total_sum = used_pixels.sum(axis=1)
        #
        # self.readings = total_sum

        # Iterative (keep for cython)
        for i in range(n_photoreceptors):

            # Vectorised
            x_len = round(xmax[i, 0] - xmin[i, 0])
            y_len = round(ymax[i, 0] - ymin[i, 0])
            x_ranges = np.linspace(xmin[i], xmax[i], x_len)
            y_ranges = np.linspace(ymin[i], ymax[i], y_len)

            y_values = (m[i] * x_ranges) + c[i]
            y_values = np.floor(y_values)
            set_1 = np.stack((x_ranges, y_values), axis=-1)

            x_values = (y_ranges - c[i]) / m[i]
            x_values = np.floor(x_values)
            set_2 = np.stack((x_values, y_ranges), axis=-1)
            full_set = np.vstack((set_1, set_2))
            full_set = full_set.reshape(-1, full_set.shape[-1]).astype(int)
            # full_set = np.unique(full_set, axis=0)

            used_pixels = masked_arena_pixels[full_set[:, 0], full_set[:, 1]]
            total_sum = used_pixels.sum(axis=0)
            self.readings[i] = total_sum

            # photoreceptor_coverage[full_set_unique[:, 0], full_set_unique[:, 1]] = 1
            # try:
            #     photoreceptor_coverage[full_set[:, 0], full_set[:, 1]] = 1
            # except IndexError:
            #     x = True

            # Iterative (keep for cython)
            # for j in range(n_lines):
            #     xrange = np.arange(xmin[i, j], xmax[i, j])
            #     yrange = np.arange(ymin[i, j], ymax[i, j])
            #
            #     yvals = (m[i, j] * xrange) + c[i, j]
            #     yvals = np.floor(yvals)
            #     set1 = np.stack((xrange, yvals), axis=-1)
            #
            #     xvals = (yrange - c[i, j]) / m[i, j]
            #     xvals = np.floor(xvals)
            #     set2 = np.stack((xvals, yrange), axis=-1)
            #     full_set = np.vstack((set1, set2))
            #     full_set = np.unique(full_set, axis=0).astype(int)
            #     photoreceptor_coverage[full_set[:, 0], full_set[:, 1]] = 1
            #     # try:
            #     #     photoreceptor_coverage[full_set[:, 0], full_set[:, 1]] = 1
            #     # except IndexError:
            #     #    x = True

            # photoreceptor_coverage = np.expand_dims(photoreceptor_coverage, 2)
            # # photoreceptor_coverage = np.repeat(photoreceptor_coverage, 3, 2)
            # weighted_points = photoreceptor_coverage * masked_arena_pixels
            # total_sum = weighted_points.sum(axis=(0, 1))

    def compute_segment_sums_line_cupy(self, masked_arena_pixels, m, c, xmin, xmax, ymin, ymax):
        n_photoreceptors = m.shape[0]  # TODO: Do only once
        x_len = cp.max(np.rint(xmax[:, 0] - xmin[:, 0]).astype(int))
        y_len = cp.max(np.rint(ymax[:, 0] - ymin[:, 0]).astype(int))

        x_ranges = cp.linspace(xmin, xmax, int(x_len))
        y_ranges = cp.linspace(ymin, ymax, int(y_len))

        y_values = (m * x_ranges) + c
        y_values = cp.floor(y_values)
        set_1 = cp.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = cp.floor(x_values)
        set_2 = cp.stack((x_values, y_ranges), axis=-1)
        full_set = cp.vstack((set_1, set_2))
        full_set = full_set.reshape(n_photoreceptors, -1, full_set.shape[-1]).astype(int)

        used_pixels = masked_arena_pixels[full_set[:, :, 0], full_set[:, :, 1]]
        total_sum = used_pixels.sum(axis=1)

        self.readings = total_sum.get()
