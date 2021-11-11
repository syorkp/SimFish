import numpy as np
from skimage.draw import line
import math
import cProfile
import time
import pstats


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
        self.photoreceptor_rf_size = 0.0014
        self.retinal_field_size = retinal_field
        self.photoreceptor_spacing = self.retinal_field_size/self.photoreceptor_num

        # Create matrix A of coordinates (w.h.3.2)
        all_points = [[[x, y] for x in range(self.width)] for y in range(self.height)]
        all_points = np.expand_dims(np.array(all_points), axis=2)
        self.rearranged_coordinates = np.repeat(all_points, repeats=3, axis=2)

        # self.profile = cProfile.Profile()
        # self.profile.enable()

    def cartesian(self, bx, by, bangle):
        x = bx + self.dist * np.cos(self.theta + bangle)
        y = (by + self.dist * np.sin(self.theta + bangle))
        return x, y

    def update_angles(self, verg_angle, retinal_field, is_left):
        # TODO: check if still needed (probably not if not using dist and theta
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        self.vis_angles = np.linspace(min_angle, max_angle, self.num_arms)
        self.dist, self.theta = np.meshgrid(self.distances, self.vis_angles)

    def show_points(self, bx, by, bangle):
        x, y = self.cartesian(bx, by, bangle)
        x = x.astype(int)
        y = y.astype(int)
        for arm in range(x.shape[0]):
            for pnt in range(x.shape[1]):
                if not (x[arm, pnt] < 0 or x[arm, pnt] >= self.width or y[arm, pnt] < 0 or y[arm, pnt] >= self.height):
                    self.board.db[y[arm, pnt], x[arm, pnt], :] = (1, 1, 1)

        [rr, cc] = line(y[0, 0], x[0, 0], y[0, 1], x[0, 1])
        good_points = np.logical_and.reduce((rr > 0, rr < self.height, cc > 0, cc < self.width))
        self.board.db[rr[good_points], cc[good_points]] = (1, 1, 1)
        [rr, cc] = line(y[-1, 0], x[-1, 0], y[-1, 1], x[-1, 1])
        good_points = np.logical_and.reduce((rr > 0, rr < self.height, cc > 0, cc < self.width))
        self.board.db[rr[good_points], cc[good_points]] = (1, 1, 1)

    def read(self, masked_arena_pixels, fish_x, fish_y, fish_angle):
        for i, channel_angle in enumerate(self.vis_angles):
            vertices = self.compute_channel_vertices([fish_x, fish_y], fish_angle, channel_angle, self.photoreceptor_rf_size)
            print(vertices)
            if len(vertices) == 3:
                segment_sum = self.sum_within_triangle(masked_arena_pixels, vertices)
            else:
                segment_sum = self.sum_within_polygon(masked_arena_pixels, vertices)
            # Segment sum by far longest point. - 99% of time.
            print(segment_sum)
            self.readings[i] = segment_sum
            # ps = pstats.Stats(self.profile)
            # ps.sort_stats("tottime")
            # ps.print_stats(20)

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

    def check_if_needs_corner(self, intersections):
        # TODO: Doesnt work
        # ix = intersections[1][0] % self.width
        # iy = intersections[1][1] % self.height
        # jx = intersections[2][0] % self.width
        # jy = intersections[2][1] % self.height
        # # if (ix != 0 and )
        # if (intersections[1][0] % self.width == 0) != (intersections[2][1] % self.height == 0) and (intersections[1][1] < 0) != (intersections[2][0] < 0):
        #     x_vals = [inters[0] for inters in intersections]
        #     y_vals = [inters[1] for inters in intersections]
        #     corner = []
        #     for i in x_vals:
        #         if i == 0 or i == self.width:
        #             corner.append(i)
        #     for i in y_vals:
        #         if i == 0 or i == self.height:
        #             corner.append(i)
        #     intersections.append(corner)
        #     return corner
        return intersections

    @staticmethod
    def get_valid_intersections(intersections, fish_position, angle):
        vector_1 = [intersections[0][0]-fish_position[0], intersections[0][1]-fish_position[1]]
        vector_2 = [intersections[1][0]-fish_position[0], intersections[1][1]-fish_position[1]]
        angle_1 = math.atan2(vector_1[1], vector_1[0])
        angle_2 = math.atan2(vector_2[1], vector_2[0])

        if angle_1 < 0:
            angle_1 += 2 * np.pi
        if angle_2 < 0:
            angle_2 += 2 * np.pi

        if round(angle_1, 2) == round(angle, 2):
            return [intersections[0]]
        elif round(angle_2, 2) == round(angle, 2):
            return [intersections[1]]
        else:
            print(f"Angle 1: {angle_1}, angle 2: {angle_2}, angle: {angle}")
            print("ERROR, INVALID ANGLE CALCULATION")

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

        intersections = self.check_if_needs_corner(intersections) # TODO: Haven't written yet.

        return intersections

    @staticmethod
    def check_cross_product(ab, bc, ca, ap, bp, cp):
        """Computes third component of cross products and returns true if they have the same sign"""
        a_cp = ab[0]*ap[1] - ab[1]*ap[0]
        b_cp = bc[0]*bp[1] - bc[1]*bp[0]
        c_cp = ca[0]*cp[1] - ca[1]*cp[0]
        return (a_cp < 0) == (b_cp < 0) == (c_cp < 0)

    def sum_within_triangle(self, masked_arena_pixels, vertices):
        # Sort vertices in ascending order of x value.
        vertices = sorted(vertices, key=lambda x: x[0])

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

        # Create matrix B of triangle vertices (w.h.3.2)
        repeating_unit = np.array([[xa, ya], [xb, yb], [xc, yc]])
        full_field = np.expand_dims(repeating_unit, axis=0)
        full_field = np.expand_dims(full_field, axis=0)
        full_field = np.repeat(full_field, self.width, 0)
        full_field = np.repeat(full_field, self.height, 1)
        # TODO: check that coordinates are correct in case of differing width and height

        # Compute C = A-B (corresponds to subtracting vertices points from each coordinate in space (w.h.3.2)
        new_vector_points = self.rearranged_coordinates - full_field

        # Flip C along final axis (so y is where x was previously) (w.h.3.2)
        new_vector_points_flipped = np.flip(new_vector_points, 3)

        # Create matrix D with repeated triangle side vectors (w.h.3.2)
        old_vector_points = [ab, bc, ca]
        old_vector_points = np.expand_dims(old_vector_points, axis=0)
        old_vector_points = np.expand_dims(old_vector_points, axis=0)
        old_vector_points = np.repeat(old_vector_points, 1500, 0)
        old_vector_points = np.repeat(old_vector_points, 1500, 1)

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

        # If points in cross product are negative, set equal to 1, else 0. (w.h.3.2)
        cross_product_less_than = (cross_product < 0) * 1
        cross_product_boolean_axis_sum = np.sum(cross_product_less_than, axis=2)

        # Compute product along final axis of each (should be 1 if all were negative)
        sector_points = ((cross_product_boolean_axis_sum == 0) | (cross_product_boolean_axis_sum == 3)) * 1

        number_enclosed = np.sum(sector_points)
        if number_enclosed == 0:
            print("Thinks is zero enclosed")

        # Expand enclosed points to dimensions in order to multiply by mask. (w.h.3)
        sector_points = np.expand_dims(sector_points, 2)
        sector_points = np.repeat(sector_points, 3, 2)

        # Multiply enclosion mask by pixel mask (w.h.3)
        weighted_points = sector_points * masked_arena_pixels

        # Sum values from entire matrix along all but final axis (3)
        total_sum = weighted_points.sum(axis=(0, 1))

        # Old method of computation
        # all_points_1 = [[x, y] for x in range(1500) for y in range(1500)]
        # points_in_triangle = []
        # for point in all_points_1:
        #     ap = [point[0]-xa, point[1]-ya]
        #     bp = [point[0]-xb, point[1]-yb]
        #     cp = [point[0]-xc, point[1]-yc]
        #
        #     if self.check_cross_product(ab, bc, ca, ap, bp, cp):
        #         points_in_triangle.append(point)
        # print(f"Pixels enclosed: NEW: {number_enclosed}, OLD: {len(points_in_triangle)} ")
        # sum = [0, 0, 0]
        # for point in points_in_triangle:
        #     sum += masked_arena_pixels[point[0], point[1], :]
        # print(f"SUM: {len(artificial_tally)}, Total sum: {number_enclosed} \n")
        return total_sum

    def sum_within_polygon(self, masked_arena_pixels, vertices):
        ...


    @staticmethod
    def remove_green_channel(board):
        """Removes the green channel, which is not used."""
        board = np.delete(board, 1, 2)
        return board

    def get_all_sectors(self, fish_position, fish_orientation):
        """Returns list of all the sectors in that part of visual field."""
        sectors = []
        for i, channel_angle in enumerate(self.vis_angles):
            vertices = self.compute_channel_vertices(fish_position, fish_orientation, channel_angle, self.photoreceptor_rf_size)
            sectors.append(vertices)
        return sectors
