import numpy as np
import cupy as cp

import skimage.draw as draw
from skimage import io

from Tools.ray_cast import rays


class NewDrawingBoard:

    def __init__(self, width, height, decay_rate):

        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.db = None
        self.erase()

        # Set of coordinates
        # xp, yp = np.meshgrid(range(self.width), range(self.height))
        # self.coordinates = np.concatenate((xp, yp), 3)

        # xp, yp = cp.meshgrid(cp.arange(self.width), cp.arange(self.height))
        self.xp, self.yp = cp.arange(self.width), cp.arange(self.height)
        # self.coordinates = cp.concatenate((xp, yp), 2)

        # self.scatter = cp.vectorize(lambda i, j, x, y: np.exp(-self.decay_rate * (((x - i) ** 2 + (y - j) ** 2) ** 0.5)))

    def scatter(self, i, j, x, y):
        return cp.exp(-self.decay_rate * (((x - i) ** 2 + (y - j) ** 2) ** 0.5))

    @staticmethod
    def apply_mask(board, mask):
        # TODO: speed up
        # new_board = np.zeros(board.shape)
        # for channel in range(board.shape[-1]):
        #     new_board[:, :, channel] = np.multiply(board[:, :, channel], mask)
        # return new_board
        mask = np.expand_dims(mask, 2)
        return board * mask

    def decay(self, fish_position):
        return np.exp(-self.decay_rate * (((fish_position[0] - i) ** 2 + (fish_position[1] - j) ** 2) ** 0.5))

    def create_scatter_mask(self, fish_position):
        """Creates the scatter mask according to the equation: I(d)=e^(-decay_rate*d), where d is distance from fish,
        computed here for all coordinates."""
        mask = np.fromfunction(
            lambda i, j: np.exp(-self.decay_rate * (((fish_position[0] - i) ** 2 + (fish_position[1] - j) ** 2) ** 0.5)),
            (self.width, self.height,),
            dtype=float)
        mask = np.expand_dims(mask, 2)
        return mask

    def create_obstruction_mask(self, fish_position):
        # TODO: Implement. What inputs does it need?
        return np.ones((self.width, self.height, 1))

    def create_luminance_mask(self):
        # TODO: implement.
        return np.ones((self.width, self.height, 1))

    def get_masked_pixels_cupy(self, fish_position):
        A = cp.array(self.db)
        L = cp.ones((self.width, self.height, 1))
        O = cp.ones((self.width, self.height, 1))
        S = self.scatter(self.xp[:, None], self.yp[:, None], fish_position[0], fish_position[1])
        return A * L * O * S

    def get_masked_pixels(self, fish_position):
        A = self.db
        L = self.create_luminance_mask()
        O = self.create_obstruction_mask(fish_position)
        S = self.create_scatter_mask(fish_position)
        return A * L * O * S
        # masked_arena = self.apply_mask(self.apply_mask(self.apply_mask(A, L), O), S)
        # return masked_arena

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

    def read_rays(self, xmat, ymat, dark_gain, light_gain, bkg_scatter, dark_col=0):
        res = rays(xmat.astype(np.int), ymat.astype(np.int), self.db, self.height, self.width, dark_gain, light_gain,
                   bkg_scatter, dark_col=dark_col)
        return res

    def read(self, xmat, ymat):
        n_arms = xmat.shape[0]
        res = np.zeros((n_arms, 3))
        for arm in range(n_arms):
            [rr, cc] = draw.line(ymat[arm, 0].astype(int), xmat[arm, 0].astype(int), ymat[arm, 1].astype(int),
                                 xmat[arm, 1].astype(int))
            prfl = self.db[rr, cc]
            # prfl = np.array(profile_line(self.db, (ymat[arm,0], xmat[arm,0]), (ymat[arm,1], xmat[arm,1]), order=0, cval=1.))
            ps = np.sum(prfl, 1)
            if len(np.nonzero(ps)[0]) > 0:
                res[arm, :] = prfl[np.nonzero(ps)[0][0], :]
            else:
                res[arm, :] = [0, 0, 0]

        # xmat_ = np.where((xmat<0) | (xmat>=self.width), 0, xmat)
        # ymat_ = np.where((ymat<0) | (ymat>=self.height), 0, ymat)
        #
        # res = self.db[ymat_, xmat_, :]
        # res[np.where((xmat<0)|xmat>=self.width)|(ymat<0)|(ymat>=self.height), :] = [1, 0, 0]
        return res

    def show(self):
        io.imshow(self.db)
        io.show()


if __name__ == "__main__":
    d = DrawingBoard(500, 500)
    d.circle((100, 200), 100, (1, 0, 0))
    d.line((50, 50), (100, 200), (0, 1, 0))
    d.show()
