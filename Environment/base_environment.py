import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale
import pymunk

from Tools.drawing_board import DrawingBoard


class BaseEnvironment:

    """A base class to represent environments, for extension to VVR and Naturalistic environment classes."""

    def __init__(self, env_variables, draw_screen):
        self.env_variables = env_variables
        self.board = DrawingBoard(self.env_variables['width'], self.env_variables['height'])  # Made change here for size of arena
        self.draw_screen = draw_screen
        self.show_all = False
        self.num_steps = 0
        self.fish = None

        if self.draw_screen:
            self.board_fig, self.ax_board = plt.subplots()
            self.board_image = plt.imshow(np.zeros((self.env_variables['height'], self.env_variables['width'], 3)))
            plt.ion()
            plt.show()

        self.dark_col = int(self.env_variables['width'] * self.env_variables['dark_light_ratio'])
        if self.dark_col == 0: # Fixes bug with left wall always being invisible.
            self.dark_col = -1

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = self.env_variables['drag']

    def readings_to_photons(self, readings):
        photons = np.random.poisson(readings * self.env_variables['photon_ratio'])
        if self.env_variables['read_noise_sigma'] > 0:
            noise = np.random.randn(readings.shape[0], readings.shape[1]) * self.env_variables['read_noise_sigma']
            photons += noise.astype(int)
        return photons

    def get_visual_inputs(self):
        left_photons = self.readings_to_photons(self.fish.left_eye.readings)
        right_photons = self.readings_to_photons(self.fish.right_eye.readings)
        left_eye = resize(np.reshape(left_photons, (1, len(self.fish.left_eye.vis_angles), 3)) * (255 / self.env_variables['photon_ratio']), (20, self.env_variables['width'] / 2 - 50))
        right_eye = resize(np.reshape(right_photons, (1, len(self.fish.right_eye.vis_angles), 3)) * (255 / self.env_variables['photon_ratio']), (20, self.env_variables['width'] / 2 - 50))
        eyes = np.hstack((left_eye, np.zeros((20, 100, 3)), right_eye))
        eyes[eyes < 0] = 0
        eyes[eyes > 255] = 255
        return eyes

    def output_frame(self, activations, internal_state, scale=0.25):
        arena = self.board.db*255.0
        arena[0, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[self.env_variables['height'] - 1, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[:, 0, 0] = np.ones(self.env_variables['height']) * 255
        arena[:, self.env_variables['width'] - 1, 0] = np.ones(self.env_variables['height']) * 255

        eyes = self.get_visual_inputs()

        frame = np.vstack((arena, np.zeros((50, self.env_variables['width'], 3)), eyes))

        this_ac = np.zeros((20, self.env_variables['width'], 3))
        this_ac[:, :, 0] = resize(internal_state, (20, self.env_variables['width']), anti_aliasing=False, order=0) * 255
        this_ac[:, :, 1] = resize(internal_state, (20, self.env_variables['width']), anti_aliasing=False, order=0) * 255
        this_ac[:, :, 2] = resize(internal_state, (20, self.env_variables['width']), anti_aliasing=False, order=0) * 255

        frame = np.vstack((frame, np.zeros((20, self.env_variables['width'], 3)), this_ac))

        if activations is not None:

            adr = [-1, 1]
            for ac in range(len(activations)):
                this_ac = np.zeros((20, self.env_variables['width'], 3))
                pos = (activations[ac] - adr[0]) / (adr[1]-adr[0])

                pos[pos < 0] = 0
                pos[pos > 1] = 1

                this_ac[:, :, 0] = resize(pos, (20, self.env_variables['width'])) * 255
                this_ac[:, :, 1] = resize(pos, (20, self.env_variables['width'])) * 255
                this_ac[:, :, 2] = resize(pos, (20, self.env_variables['width'])) * 255

                frame = np.vstack((frame, np.zeros((20, self.env_variables['width'], 3)), this_ac))

        frame = rescale(frame, scale, multichannel=True, anti_aliasing=True)
        return frame

    def touch_edge(self, arbiter, space, data):
        self.fish.body.velocity = (0, 0)
        if self.fish.body.angle < np.pi:
            self.fish.body.angle += np.pi
        else:
            self.fish.body.angle -= np.pi
        self.fish.body.apply_impulse_at_local_point((20, 0))

        self.fish.touched_edge = True
        return True


