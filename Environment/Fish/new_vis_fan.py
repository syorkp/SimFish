import numpy as np
from skimage.draw import line
import math


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
        self.photoreceptor_rf_size = 0.14
        self.retinal_field_size = retinal_field
        self.photoreceptor_spacing = self.retinal_field_size/self.photoreceptor_num

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
        for channel_angle in self.vis_angles:
            vertices = self.compute_channel_vertices([fish_x, fish_y], fish_angle, channel_angle, self.photoreceptor_rf_size)
            if len(vertices) == 3:
                segment_sum = self.sum_within_triangle(masked_arena_pixels, vertices)
            else:
                segment_sum = self.sum_within_polygon(masked_arena_pixels, vertices)

            # Compute angle of field
            # Compute channel vertices
            # Sum contents in masked_arena_pixels.
            # Add to readings index.
            ...

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
        return intersections

    def get_valid_intersections(self, intersections, fish_position, angle):
        vector_1 = [intersections[0][0]-fish_position[0], intersections[0][1]-fish_position[1]]
        vector_2 = [intersections[1][0]-fish_position[0], intersections[1][1]-fish_position[1]]
        angle_1 = math.atan2(vector_1[1], vector_1[0])
        angle_2 = math.atan2(vector_2[1], vector_2[0])
        if angle_1 == angle:
            return [intersections[0]]
        elif angle_2 == angle:
            return [intersections[1]]
        else:
            print("ERROR, INVALID ANGLE CALCULATION")

    def convert_angle(self, absolute_channel_theta, channel_rf_size):
        absolute_channel_theta = -absolute_channel_theta

        # TODO: Change to one system (no negatives).
        angle_a = absolute_channel_theta - (channel_rf_size/2)
        angle_b = absolute_channel_theta + (channel_rf_size/2)

    def compute_channel_vertices(self, fish_position, fish_angle, channel_orientation, channel_rf_size):
        # TODO: Ensure fish posititon in interretinal point
        absolute_channel_theta = fish_angle + channel_orientation
        angle_a, angle_b = self.convert_angle(absolute_channel_theta)

        m = np.tan([-angle_a, -angle_b])  # Need to transform orientations to negative before conversion.
        c_a = fish_position[1] - m[0]*fish_position[0]
        c_b = fish_position[1] - m[1]*fish_position[0]
        intersections_a = self.find_wall_intersections(m[0], c_a)
        intersections_b = self.find_wall_intersections(m[1], c_b)

        # Split so uses correct coordinates for vis direction
        intersections = []
        intersections += self.get_valid_intersections(intersections_a, fish_position, -angle_a)
        intersections += self.get_valid_intersections(intersections_b, fish_position, -angle_b)
        intersections = self.check_if_needs_corner(intersections)
        return [[0, 0], [0, 50], [50, 50]]

    @staticmethod
    def check_cross_product(ab, bc, ca, ap, bp, cp):
        """Computes third component of cross products and returns true if they have the same sign"""
        a_cp = ab[0]*ap[1] - ab[1]*ap[0]
        b_cp = bc[0]*bp[1] - bc[1]*bp[0]
        c_cp = ca[0]*cp[1] - ca[1]*cp[0]
        return (a_cp < 0) == (b_cp < 0) == (c_cp < 0)

    def sum_within_triangle(self, masked_arena_pixels, vertices):
        # Sort vertices
        vertices = sorted(vertices, key=lambda x: x[0])
        xa = vertices[0][0]
        xb = vertices[1][0]
        xc = vertices[2][0]
        ya = vertices[0][1]
        yb = vertices[1][1]
        yc = vertices[2][1]

        ab = [xb-xa, yb-ya]
        bc = [xc-xb, yc-yb]
        ca = [xa-xc, ya-yc]

        # # Compute coefficients of equations of lines containing segments.
        # A = y2 - y1
        # B = x1 - y1
        # C = (x2 * y1) - (x1 * y2)

        all_points = [[x, y] for x in range(1500) for y in range(1500)]
        points_in_triangle = []

        for point in all_points:
            ap = [point[0]-xa, point[1]-ya]
            bp = [point[0]-xb, point[1]-yb]
            cp = [point[0]-xc, point[1]-yc]

            if self.check_cross_product(ab, bc, ca, ap, bp, cp):
                points_in_triangle.append(point)

        sum = [0, 0, 0]
        for point in points_in_triangle:
            sum += masked_arena_pixels[point[0], point[1], :]
        x = True




    def sum_within_polygon(self, masked_arena_pixels, vertices):
        ...


    @staticmethod
    def remove_green_channel(board):
        """Removes the green channel, which is not used."""
        board = np.delete(board, 1, 2)
        return board


