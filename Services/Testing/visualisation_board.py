import numpy as np
import math
import matplotlib.pyplot as plt
from time import time

import skimage.draw as draw
from skimage import io


class VisualisationBoard:

    def __init__(self, width, height, prey_size=4,
                 predator_size=100, visible_scatter=0.3, background_grating_frequency=50, dark_light_ratio=0.0,
                 dark_gain=0.01, light_gain=1.0, light_gradient=200, max_visual_distance=1500, show_background=True):

        self.chosen_math_library = np

        self.width = width
        self.height = height
        self.light_gain = light_gain
        self.light_gradient = light_gradient
        self.db = None
        self.db_visualisation = None
        self.base_db_illuminated = self.get_base_arena(visible_scatter)
        self.background_grating = self.get_background_grating(background_grating_frequency)

        self.prey_size = prey_size * 2
        self.prey_radius = prey_size
        self.predator_size = predator_size * 2
        self.predator_radius = predator_size
        self.max_visual_distance = max_visual_distance

        # For debugging purposes
        self.mask_buffer_time_point = None

        # For obstruction mask (reset each time is called).
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

    def reset(self):
        """To be called at start of episode"""
        self.background_grating = self.get_background_grating(0)

    def erase_visualisation(self, bkg=0.3):
        if bkg == 0:
            db = self.chosen_math_library.zeros((self.height, self.width, 3), dtype=np.double)
        else:
            db = (self.chosen_math_library.ones((self.height, self.width, 3), dtype=np.double) * bkg)
        db[1:2, :] = self.chosen_math_library.array([1, 0, 0])
        db[self.width - 2:self.width - 1, :] = self.chosen_math_library.array([1, 0, 0])
        db[:, 1:2] = self.chosen_math_library.array([1, 0, 0])
        db[:, self.height - 2:self.height - 1] = self.chosen_math_library.array([1, 0, 0])
        self.db_visualisation = db

        if self.show_background:
            self.db_visualisation += self.chosen_math_library.concatenate((self.background_grating/10,
                                                                           self.background_grating/10,
                                                                           self.chosen_math_library.zeros(self.background_grating.shape)), axis=2)

    def get_base_arena(self, bkg=0.0):
        if bkg == 0:
            db = self.chosen_math_library.zeros((self.height, self.width, 3), dtype=np.double)
        else:
            db = (self.chosen_math_library.ones((self.height, self.width, 3), dtype=np.double) * bkg) / self.light_gain
        db[1:2, :] = self.chosen_math_library.array([1, 0, 0])
        db[self.width - 2:self.width - 1, :] = self.chosen_math_library.array([1, 0, 0])
        db[:, 1:2] = self.chosen_math_library.array([1, 0, 0])
        db[:, self.height - 2:self.height - 1] = self.chosen_math_library.array([1, 0, 0])
        return db

    def draw_walls(self):
        self.db[0:2, :] = [1, 0, 0]
        self.db[self.width - 1, :] = [1, 0, 0]
        self.db[:, 0] = [1, 0, 0]
        self.db[:, self.height - 1] = [1, 0, 0]

    def apply_light(self, dark_col, dark_gain, light_gain):
        if dark_col < 0:
            dark_col = 0

        if self.light_gradient > 0 and dark_col > 0:
            gradient = self.chosen_math_library.linspace(1, 2, self.light_gradient)
            gradient = self.chosen_math_library.expand_dims(gradient, 0)
            gradient = self.chosen_math_library.repeat(gradient, self.height, 0)
            gradient = self.chosen_math_library.expand_dims(gradient, 2)
            self.db_visualisation[:, int(dark_col-(self.light_gradient/2)):int(dark_col+(self.light_gradient/2))] *= gradient
            self.db_visualisation[:, :int(dark_col-(self.light_gradient/2))] *= 1
            self.db_visualisation[:, int(dark_col+(self.light_gradient/2)):] *= 2
        else:
            self.db_visualisation[:, :dark_col] *= 1
            self.db_visualisation[:, dark_col:] *= 2

    def circle(self, center, rad, color, visualisation=False):
        rr, cc = draw.circle(center[1], center[0], rad, self.db_visualisation.shape)
        self.db_visualisation[rr, cc, :] = color

    def show_salt_location(self, location):
        rr, cc = draw.circle(location[1], location[0], 10, self.db_visualisation.shape)
        self.db_visualisation[rr, cc, :] = (1, 0, 0)

    def tail(self, head, left, right, tip, color, visualisation):
        tail_coordinates = np.array((head, left, tip, right))
        rr, cc = draw.polygon(tail_coordinates[:, 1], tail_coordinates[:, 0], self.db_visualisation.shape)
        self.db_visualisation[rr, cc, :] = color

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

    def show(self):
        if self.using_gpu:
            io.imshow(self.db.get())
        else:
            io.imshow(self.db)

        io.show()


if __name__ == "__main__":
    d = NewDrawingBoard(500, 500)
    d.circle((100, 200), 100, (1, 0, 0))
    d.line((50, 50), (100, 200), (0, 1, 0))
    d.show()