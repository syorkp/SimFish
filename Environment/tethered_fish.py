import numpy as np
import pymunk

from Environment.vis_fan import VisFan


class TetheredFish:

    """
    Created to simplify the SimState class, while making it easier to have environments with multiple agents in future.
    """

    def __init__(self, board, env_variables, dark_col):
        # TODO: See if can just inherit from normal fish then overwrite the action method.
        inertia = pymunk.moment_for_circle(env_variables['fish_mass'], 0, env_variables['fish_size'], (0, 0))
        self.env_variables = env_variables
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

    def take_action(self, action):
        if action == 0:  # Swim forward
            reward = -self.env_variables['forward_swim_cost']
            self.shape.color = (0, 1, 0)
        elif action == 1:  # Turn right
            reward = -self.env_variables['routine_turn_cost']
            self.shape.color = (0, 1, 0)
        elif action == 2:   # Turn left
            reward = -self.env_variables['routine_turn_cost']
            self.shape.color = (0, 1, 0)
        elif action == 3:   # Capture
            reward = -self.env_variables['capture_swim_cost']
            self.shape.color = [1, 0, 1]
            self.making_capture = True
        elif action == 4:  # j turn right
            reward = -self.env_variables['j_turn_cost']
            self.shape.color = [1, 1, 1]
        elif action == 5:  # j turn left
            reward = -self.env_variables['j_turn_cost']
            self.shape.color = [1, 1, 1]
        elif action == 6:   # do nothing:
            reward = -self.env_variables['rest_cost']
        else:
            reward = None
            print("Invalid action given")

        # elif action == 6: #converge eyes. Be sure to update below with fish.[]
        #     self.verg_angle = 77 * (np.pi / 180)
        #     self.left_eye.update_angles(self.verg_angle, self.retinal_field, True)
        #     self.right_eye.update_angles(self.verg_angle, self.retinal_field, False)
        #     self.conv_state = 1

        # elif action == 7: #diverge eyes
        #     self.verg_angle = 25 * (np.pi / 180)
        #     self.left_eye.update_angles(self.verg_angle, self.retinal_field, True)
        #     self.right_eye.update_angles(self.verg_angle, self.retinal_field, False)
        #     self.conv_state = 0
        return reward
