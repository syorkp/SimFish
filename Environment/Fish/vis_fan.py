import numpy as np
from skimage.draw import line


class VisFan:

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

    def read(self, bx, by, bangle):
        x, y = self.cartesian(bx, by, bangle)
        self.readings = self.board.read_rays(x, y, self.dark_gain, self.light_gain, self.bkg_scatter, self.dark_col)
