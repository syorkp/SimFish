import numpy as np
import cupy as cp
import math
import matplotlib.pyplot as plt

import skimage.draw as draw
from skimage import io


class NewDrawingBoard:

    def __init__(self, width, height, decay_rate, photoreceptor_rf_size, using_gpu, prey_size=4, predator_size=100):

        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.photoreceptor_rf_size = photoreceptor_rf_size
        self.db = None
        self.erase()

        self.prey_size = prey_size * 2
        self.prey_radius = prey_size
        self.predator_size = predator_size * 2
        self.predator_radius = predator_size

        if using_gpu:
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

        self.xp, self.yp = self.chosen_math_library.arange(self.width), self.chosen_math_library.arange(self.height)

    def scatter(self, i, j, x, y):
        """Computes general scatter, but incorporates effect of implicit scatter from line spread."""
        positional_mask = (((x - i) ** 2 + (y - j) ** 2) ** 0.5)
        desired_scatter = self.chosen_math_library.exp(-self.decay_rate * positional_mask)
        implicit_scatter = self.chosen_math_library.sin(self.photoreceptor_rf_size) * positional_mask
        implicit_scatter[implicit_scatter < 1] = 1
        adjusted_scatter = desired_scatter * implicit_scatter
        adjusted_scatter = self.chosen_math_library.expand_dims(adjusted_scatter, 2)
        return adjusted_scatter

    def create_obstruction_mask_lines_mixed(self, fish_position, prey_locations, predator_locations):
        # In future, should consider way to minimise number of lines to draw by using required number specific to each.
        fish_position = np.array(fish_position)

        # Compute all angles for prey features
        prey_locations = np.array(prey_locations)

        # Compute prey positions relative to fish
        prey_relative_positions = prey_locations-fish_position

        # Compute distances of prey from fish.
        prey_distances = (prey_relative_positions[:, 0] ** 2 + prey_relative_positions[:, 1] ** 2) ** 0.5

        # Compute angular size of prey from fish position.
        prey_half_angular_size = np.arctan(self.prey_radius / prey_distances)

        # Compute angle between fish and prey - where horizontal line is 0, and positive values are in upper field.
        prey_angles = np.arctan(prey_relative_positions[:, 1]/prey_relative_positions[:, 0])

        prey_angles = np.expand_dims(prey_angles, 1)
        prey_angles = np.repeat(prey_angles, 2, 1)

        # From prey angular sizes, compute angles of edges of prey.
        prey_rf_offsets = np.expand_dims(prey_half_angular_size, 1)
        prey_rf_offsets = np.repeat(prey_rf_offsets, 2, 1)
        prey_rf_offsets = prey_rf_offsets * np.array([-1, 1])
        prey_extremities = prey_angles + prey_rf_offsets

        # Number of lines to project through prey or predators, determined by width, height, and size of features.
        n_lines_prey = self.compute_n(np.max(prey_half_angular_size)*2)

        # Create array of angles between prey extremities to form lines.
        interpolated_line_angles = np.linspace(prey_extremities[:, 0], prey_extremities[:, 1], n_lines_prey).flatten()

        # Computing how far along each line prey are.
        # prey_distance_along = (prey_distances ** 2 + self.prey_radius ** 2) ** 0.5
        prey_distance_along = prey_distances + self.prey_radius
        prey_distance_along = np.expand_dims(prey_distance_along, 1)
        prey_distance_along = np.repeat(prey_distance_along, n_lines_prey, 1)
        prey_distance_along = np.swapaxes(prey_distance_along, 0, 1).flatten()
        distance_along = prey_distance_along

        expanded_prey_locations = np.tile(prey_locations, (n_lines_prey, 1))
        prey_on_left = (expanded_prey_locations[:, 0] < fish_position[0]) * np.pi

        # Compute all angles for predator features.
        predator_locations = np.array(predator_locations)
        n_predators = len(predator_locations)

        if n_predators > 0:
            predator_relative_positions = predator_locations-fish_position
            predator_distances = (predator_relative_positions[:, 0] ** 2 + predator_relative_positions[:, 1] ** 2) ** 0.5
            predator_half_angular_size = np.arctan(self.predator_radius / predator_distances)

            predator_angles = np.arctan(predator_relative_positions[:, 1]/predator_relative_positions[:, 0])
            predator_angles_expanded = np.expand_dims(predator_angles, 1)
            predator_angles_expanded = np.repeat(predator_angles_expanded, 2, 1)

            predator_rf_offsets = np.expand_dims(predator_half_angular_size, 1)
            predator_rf_offsets = np.repeat(predator_rf_offsets, 2, 1)
            predator_rf_offsets = predator_rf_offsets * np.array([-1, 1])
            predator_extremities = predator_angles_expanded + predator_rf_offsets

            # Number of lines to project through prey or predators, determined by width, height, and size of features.
            n_lines_predator = self.compute_n(np.max(predator_half_angular_size) * 2)
            predator_interpolated_line_angles = np.linspace(predator_extremities[:, 0], predator_extremities[:, 1],
                                                            n_lines_predator).flatten()

            # Combine prey and predator line angles.
            interpolated_line_angles = np.concatenate((interpolated_line_angles, predator_interpolated_line_angles), axis=0)

            # Computing how far along each line predators are.
            predator_distance_along = (predator_distances ** 2 + self.predator_radius ** 2) ** 0.5
            predator_distance_along = np.expand_dims(predator_distance_along, 1)
            predator_distance_along = np.repeat(predator_distance_along, int(n_lines_predator), 1)
            predator_distance_along = np.swapaxes(predator_distance_along, 0, 1).flatten()
            predator_distance_along = predator_distance_along + self.predator_radius
            distance_along = np.concatenate((distance_along, predator_distance_along), axis=0)

            expanded_predator_locations = np.tile(predator_locations, (n_lines_predator, 1))  # np.concatenate((prey_locations, prey_locations), axis=0)
            predators_on_left = (expanded_predator_locations[:, 0] < fish_position[0]) * np.pi
            prey_on_left = np.concatenate((prey_on_left, predators_on_left), 0)

        total_lines = interpolated_line_angles.shape[0]

        interpolated_line_angles_scaling = (interpolated_line_angles // (np.pi * 2)) * np.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        # below_range = (interpolated_line_angles <= 0) * np.pi * 2
        # interpolated_line_angles = interpolated_line_angles + below_range
        # above_range = (interpolated_line_angles > np.pi * 2) * - np.pi*2
        # interpolated_line_angles = interpolated_line_angles + above_range

        # Compute m using tan (N_obj x n)
        m = np.tan(interpolated_line_angles)

        # Compute c (N_obj*n)
        c = -m * fish_position[0]
        c = c + fish_position[1]

        # Compute components of intersections (N_obj*n x 4)
        c_exp = np.expand_dims(c, 1)
        c_exp = np.repeat(c_exp, 4, 1)

        multiplication_matrix_unit = np.array([-1, 1, -1, 1])
        multiplication_matrix = np.tile(multiplication_matrix_unit, (total_lines, 1))

        addition_matrix_unit = np.array([0, 0, self.height-1, self.width-1])
        addition_matrix = np.tile(addition_matrix_unit, (total_lines, 1))

        # TODO: Can compute bits once, then clip them with total_lines number.
        mul1 = np.array([0, 0, 0, 1])
        mul1_full = np.tile(mul1, (total_lines, 1))
        m_mul = np.expand_dims(m, 1)
        full_m = np.repeat(m_mul, 4, 1)
        m_mul = full_m * mul1_full
        m_mul[:, :3] = 1
        addition_matrix = addition_matrix * m_mul
        division_matrix = full_m
        division_matrix[:, 1] = 1
        division_matrix[:, 3] = 1

        intersection_components = ((c_exp * multiplication_matrix) + addition_matrix)/division_matrix

        mul_for_hypothetical = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        mul_for_hypothetical = np.tile(mul_for_hypothetical, (total_lines, 1, 1))
        add_for_hypothetical = np.array([[0, 0], [0, 0], [0, self.width-1], [self.height-1, 0]])
        add_for_hypothetical = np.tile(add_for_hypothetical, (total_lines, 1, 1))

        intersection_coordinates = np.expand_dims(intersection_components, 2)
        intersection_coordinates = np.repeat(intersection_coordinates, 2, 2)
        intersection_coordinates = (intersection_coordinates * mul_for_hypothetical) + add_for_hypothetical

        # Compute possible intersections (N_obj n 2 x 2 x 2)
        conditional_tiled = np.array([self.width-1, self.height-1, self.width-1, self.height-1])
        conditional_tiled = np.tile(conditional_tiled, (total_lines, 1))
        valid_points_ls = (intersection_components > 0) * 1
        valid_points_more = (intersection_components < conditional_tiled) * 1
        valid_points = valid_points_more * valid_points_ls
        valid_intersection_coordinates = intersection_coordinates[valid_points == 1]

        # Get intersections (N_obj x 2)
        possible_vectors = valid_intersection_coordinates - fish_position
        angles = np.arctan2(possible_vectors[:, 1], possible_vectors[:, 0])

        # Make sure angles are in correct range. TODO: be aware might need to repeat multiple times later
        # below_range = (angles <= 0) * np.pi * 2
        # angles = angles + below_range
        # above_range = (angles > (np.pi * 2)) * (-np.pi*2)
        # angles = angles + above_range
        angles_scaling = (angles // (np.pi * 2)) * np.pi * -2
        angles = angles + angles_scaling

        angles = np.round(angles, 2)

        # Add adjustment for features appearing in left of visual field (needed because of angles)
        interpolated_line_angles = interpolated_line_angles + prey_on_left
        # below_range = (interpolated_line_angles <= 0) * np.pi * 2
        # interpolated_line_angles = interpolated_line_angles + below_range
        # above_range = (interpolated_line_angles > (np.pi * 2)) * (-np.pi*2)
        # interpolated_line_angles = interpolated_line_angles + above_range
        interpolated_line_angles_scaling = (interpolated_line_angles // (np.pi * 2)) * np.pi * -2
        interpolated_line_angles = interpolated_line_angles + interpolated_line_angles_scaling

        channel_angles_surrounding = np.round(interpolated_line_angles, 2)
        channel_angles_surrounding = np.expand_dims(channel_angles_surrounding, 1)
        channel_angles_surrounding = np.repeat(channel_angles_surrounding, 2, 1).flatten()

        same_values = (angles == channel_angles_surrounding) * 1
        selected_intersections = valid_intersection_coordinates[same_values == 1]

        # TODO: replace eye position with computation of vertices
        # Finding coordinates of object extremities.
        proj_vector = selected_intersections - fish_position
        proj_distance = (proj_vector[:, 0] ** 2 + proj_vector[:, 1] ** 2) ** 0.5  # Only really need to do for one as is same distance along.

        try:
            fraction_along = distance_along/proj_distance
        except ValueError:
            x = True
            print("Value error")
        if np.any(fraction_along > 1):
            print("Error")
            x = True
        fraction_along = np.expand_dims(fraction_along, 1)
        fraction_along = np.repeat(fraction_along, 2, 1)

        points_on_features = proj_vector * fraction_along
        points_on_features = fish_position + points_on_features
        points_on_features = np.expand_dims(points_on_features, 1)

        selected_intersections = np.reshape(selected_intersections, (total_lines, 1, 2))

        vertices = np.concatenate((selected_intersections, points_on_features), 1)
        vertices_xvals = vertices[:, :, 0]
        vertices_yvals = vertices[:, :, 1]

        # INTERPOLATION
        # TODO: Probably faster way of doing below...
        min_x = np.min(vertices_xvals, axis=1)
        max_x = np.max(vertices_xvals, axis=1)
        min_y = np.min(vertices_yvals, axis=1)
        max_y = np.max(vertices_yvals, axis=1)

        # SEGMENT COMPUTATION  # TODO: Make sure this is enough to cover span.CHANGD HERE
        x_lens = np.rint(max_x - min_x)
        y_lens = np.rint(max_y - min_y)

        x_len = np.max(x_lens)
        y_len = np.max(y_lens)

        x_ranges = np.linspace(min_x, max_x, int(x_len))
        y_ranges = np.linspace(min_y, max_y, int(y_len))

        y_values = (m * x_ranges) + c
        y_values = np.floor(y_values)
        set_1 = np.stack((x_ranges, y_values), axis=-1)
        x_values = (y_ranges - c) / m
        x_values = np.floor(x_values)
        set_2 = np.stack((x_values, y_ranges), axis=-1)
        full_set = np.vstack((set_1, set_2)).astype(int)
        full_set = self.chosen_math_library.array(full_set)

        full_set = full_set.reshape(-1, 2)
        mask = self.chosen_math_library.ones((1500, 1500), dtype=int)

        mask[full_set[:, 1], full_set[:, 0]] = 0  # NOTE: Inverting x and y to match standard in program.

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

        mask = self.chosen_math_library.expand_dims(mask, 2)

        return mask

    def create_luminance_mask(self):
        # TODO: implement.
        return np.ones((self.width, self.height, 1))

    def get_masked_pixels(self, fish_position, prey_locations, predator_locations, visualise_mask=False):
        visualise_mask = False
        if visualise_mask:
            AV = self.chosen_math_library.array(self.db)
        A = self.chosen_math_library.array(np.delete(self.db, 1, axis=2))
        L = self.chosen_math_library.ones((self.width, self.height, 1))
        O = self.create_obstruction_mask_lines_mixed(fish_position, prey_locations, predator_locations)
        # O = self.chosen_math_library.ones((self.width, self.height, 1))
        S = self.scatter(self.xp[:, None], self.yp[None, :], fish_position[1], fish_position[0])

        if visualise_mask:
            # plt.imshow(AV)
            # plt.show()
            plt.imshow(O)
            plt.show()
            # plt.imshow(S)
            # plt.show()
            # G = AV * L * O * S
            # plt.imshow(G)
            # plt.show()

        return A * L * O * S

    def compute_n(self, angular_size, max_separation=1):
        max_dist = (self.width**2 + self.height**2)**0.5
        theta_separation = math.asin(max_separation/max_dist)
        n = (angular_size/theta_separation)/2
        return int(n)

    def erase(self, bkg=0):
        if bkg == 0:
            self.db = np.zeros((self.height, self.width, 3), dtype=np.double)
        else:
            self.db = np.ones((self.height, self.width, 3), dtype=np.double) * bkg
        self.draw_walls()

    def draw_walls(self):
        self.db[0:2, :] = [1, 0, 0]
        self.db[self.width-1, :] = [1, 0, 0]
        self.db[:, 0] = [1, 0, 0]
        self.db[:, self.height-1] = [1, 0, 0]

    def apply_light(self, dark_col, dark_gain, light_gain):
        self.db[:, :dark_col] *= dark_gain
        self.db[:, dark_col:] *= light_gain

    def circle(self, center, rad, color):
        rr, cc = draw.circle(center[1], center[0], rad, self.db.shape)
        self.db[rr, cc, :] = color

    def tail(self, head, left, right, tip, color):
        tail_coordinates = np.array((head, left, tip, right))
        rr, cc = draw.polygon(tail_coordinates[:, 1], tail_coordinates[:, 0], self.db.shape)
        self.db[rr, cc, :] = color

    def fish_shape(self, mouth_centre, mouth_rad, head_rad, tail_length, mouth_colour, body_colour, angle):
        offset = np.pi / 2
        angle += offset
        angle = -angle
        self.circle(mouth_centre, mouth_rad, mouth_colour)  # For the mouth.
        dx1, dy1 = head_rad * np.sin(angle), head_rad * np.cos(angle)
        head_centre = (mouth_centre[0] + dx1,
                       mouth_centre[1] + dy1)
        self.circle(head_centre, head_rad, body_colour)
        dx2, dy2 = -1 * dy1, dx1
        left_flank = (head_centre[0] + dx2,
                      head_centre[1] + dy2)
        right_flank = (head_centre[0] - dx2,
                       head_centre[1] - dy2)
        tip = (mouth_centre[0] + (tail_length + head_rad) * np.sin(angle),
               mouth_centre[1] + (tail_length + head_rad) * np.cos(angle))
        self.tail(head_centre, left_flank, right_flank, tip, body_colour)

    def create_screen(self, fish_position, distance, colour):
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 10)
        self.db[rr, cc, :] = colour
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 9)
        self.db[rr, cc, :] = colour
        rr, cc = draw.circle_perimeter(int(fish_position[0]), int(fish_position[1]), distance - 8)
        self.db[rr, cc, :] = colour

    def vegetation(self, vertex, edge_size, color):
        coordinates = np.array(((vertex[1], vertex[0]),
                                (vertex[1], vertex[0] + edge_size),
                                (vertex[1] + edge_size / 2, vertex[0] + edge_size - edge_size / 3),
                                (vertex[1] + edge_size, vertex[0] + edge_size),
                                (vertex[1] + edge_size, vertex[0]),
                                (vertex[1] + edge_size / 2, vertex[0] + edge_size / 3)))

        rr, cc = draw.polygon(coordinates[:, 0], coordinates[:, 1], self.db.shape)
        self.db[rr, cc, :] = color

    @staticmethod
    def multi_circles(cx, cy, rad):
        rr, cc = draw.circle(0, 0, rad)
        rrs = np.tile(rr, (len(cy), 1)) + np.tile(np.reshape(cy, (len(cy), 1)), (1, len(rr)))
        ccs = np.tile(cc, (len(cx), 1)) + np.tile(np.reshape(cx, (len(cx), 1)), (1, len(cc)))
        return rrs, ccs

    def line(self, p1, p2, color):
        rr, cc = draw.line(p1[1], p1[0], p2[1], p2[0])
        self.db[rr, cc, :] = color

    def get_size(self):
        return self.width, self.height

    def show(self):
        io.imshow(self.db)
        io.show()


if __name__ == "__main__":
    d = NewDrawingBoard(500, 500)
    d.circle((100, 200), 100, (1, 0, 0))
    d.line((50, 50), (100, 200), (0, 1, 0))
    d.show()
