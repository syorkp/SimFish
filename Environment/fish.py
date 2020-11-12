import numpy as np
import pymunk

from Environment.vis_fan import VisFan


class Fish:

    """
    Created to simplify the SimState class, while making it easier to have environments with multiple agents in future.
    """

    def __init__(self, board, env_variables, dark_col):
        inertia = pymunk.moment_for_circle(env_variables['fish_mass'], 0, env_variables['fish_size'], (0, 0))
        self.body = pymunk.Body(1, inertia)
        self.shape = pymunk.Circle(self.body, env_variables['fish_size'])
        self.shape.color = (0, 1, 0)
        self.shape.elasticity = 1.0
        self.shape.collision_type = 3

        self.verg_angle = env_variables['eyes_verg_angle'] * (np.pi / 180)
        self.retinal_field = env_variables['visual_field'] * (np.pi / 180)
        self.conv_state = 0

        self.left_eye = VisFan(board, self.verg_angle, self.retinal_field, True,
                               env_variables['num_photoreceptors'], env_variables['min_vis_dist'],
                               env_variables['max_vis_dist'], env_variables['dark_gain'],
                               env_variables['light_gain'], env_variables['bkg_scatter'], dark_col)

        self.right_eye = VisFan(board, self.verg_angle, self.retinal_field, False,
                                env_variables['num_photoreceptors'], env_variables['min_vis_dist'],
                                env_variables['max_vis_dist'], env_variables['dark_gain'],
                                env_variables['light_gain'], env_variables['bkg_scatter'], dark_col)

        self.hungry = 0
        self.prey_consumed = False
        self.touched_edge = False
        self.touched_predator = False
        self.making_capture = False
