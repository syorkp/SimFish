import numpy as np
import pymunk
from skimage.transform import resize, rescale

from Environment.Fish.vis_fan import VisFan
from Environment.Action_Space.draw_angle_dist import draw_angle_dist


class Fish:

    """
    Created to simplify the SimState class, while making it easier to have environments with multiple agents in future.
    """
    def __init__(self, board, env_variables, dark_col, realistic_bouts, fish_mass=None):

        # For the purpose of producing a calibration curve.
        if fish_mass is None:
            inertia = pymunk.moment_for_circle(env_variables['fish_mass'], 0, env_variables['fish_head_size'], (0, 0))
        else:
            inertia = pymunk.moment_for_circle(fish_mass, 0, env_variables['fish_mouth_size'], (0, 0))

        self.env_variables = env_variables
        self.body = pymunk.Body(1, inertia)

        self.realistic_bouts = realistic_bouts

        # Mouth
        self.mouth = pymunk.Circle(self.body, env_variables['fish_mouth_size'], offset=(0, 0))
        self.mouth.color = (1, 0, 1)
        self.mouth.elasticity = 1.0
        self.mouth.collision_type = 3

        # Head
        self.head = pymunk.Circle(self.body, env_variables['fish_head_size'], offset=(-env_variables['fish_head_size'], 0))
        self.head.color = (0, 1, 0)
        self.head.elasticity = 1.0
        self.head.collision_type = 3

        # # Tail
        tail_coordinates = ((-env_variables['fish_head_size'], 0),
                            (-env_variables['fish_head_size'], - env_variables['fish_head_size']),
                            (-env_variables['fish_head_size'] - env_variables['fish_tail_length'], 0),
                            (-env_variables['fish_head_size'], env_variables['fish_head_size']))
        self.tail = pymunk.Poly(self.body, tail_coordinates)
        self.tail.color = (0, 1, 0)
        self.tail.elasticity = 1.0
        self.tail.collision_type = 6

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
        self.prev_action_impulse = 0

    def take_action(self, action):
        if self.realistic_bouts:
            return self.take_realistic_action(action)
        else:
            return self.take_basic_action(action)

    def take_basic_action(self, action):
        """Original version"""
        if action == 0:  # Swim forward
            reward = -self.env_variables['forward_swim_cost']
            self.body.apply_impulse_at_local_point((self.env_variables['forward_swim_impulse'], 0))
            self.head.color = (0, 1, 0)
        elif action == 1:  # Turn right
            reward = -self.env_variables['routine_turn_cost']
            self.body.angle += self.env_variables['routine_turn_dir_change']
            self.body.apply_impulse_at_local_point((self.env_variables['routine_turn_impulse'], 0))
            self.head.color = (0, 1, 0)
        elif action == 2:   # Turn left
            reward = -self.env_variables['routine_turn_cost']
            self.body.angle -= self.env_variables['routine_turn_dir_change']
            self.body.apply_impulse_at_local_point((self.env_variables['routine_turn_impulse'], 0))
            self.head.color = (0, 1, 0)
        elif action == 3:   # Capture
            reward = -self.env_variables['capture_swim_cost']
            self.body.apply_impulse_at_local_point((self.env_variables['capture_swim_impulse'], 0))
            self.head.color = [1, 0, 1]
            self.making_capture = True
        elif action == 4:  # j turn right
            reward = -self.env_variables['j_turn_cost']
            self.body.angle += self.env_variables['j_turn_dir_change']
            self.body.apply_impulse_at_local_point((self.env_variables['j_turn_impulse'], 0))
            self.head.color = [1, 1, 1]
        elif action == 5:  # j turn left
            reward = -self.env_variables['j_turn_cost']
            self.body.angle -= self.env_variables['j_turn_dir_change']
            self.body.apply_impulse_at_local_point((self.env_variables['j_turn_impulse'], 0))
            self.head.color = [1, 1, 1]
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

    def calculate_impulse(self, distance):
        """
        Uses the derived distance-mass-impulse relationship to convert an input distance (in mm) to impulse
        (arbitrary units).
        :param distance:
        :return:
        """
        return (distance*10 - (0.004644*self.env_variables['fish_mass'] + 0.081417))/1.771548

    @staticmethod
    def calculate_angle_cost(angle, distance):
        """
        So far, a fairly arbitrary equation to calculate action cost from distance moved and angle changed.
        cost = 0.05(angle change) + 1.5(distance moved)
        :return:
        """
        return 0.05 * abs(angle) + 1.5 * distance

    def take_realistic_action(self, action):
        if action == 0:  # Slow2
            angle_change, distance = draw_angle_dist(8)
            reward = -self.calculate_angle_cost(angle_change, distance)
            self.body.angle += np.random.choice([-angle_change, angle_change])
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 1:  # RT right
            angle_change, distance = draw_angle_dist(7)
            reward = -self.calculate_angle_cost(angle_change, distance)
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 2:  # RT left
            angle_change, distance = draw_angle_dist(7)
            reward = -self.calculate_angle_cost(angle_change, distance)
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 3:  # Short capture swim
            angle_change, distance = draw_angle_dist(0)
            reward = -self.calculate_angle_cost(angle_change, distance) - self.env_variables['capture_swim_extra_cost']
            self.body.angle += np.random.choice([-angle_change, angle_change])
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 1]
            self.making_capture = True

        elif action == 4:  # j turn right
            angle_change, distance = draw_angle_dist(4)
            reward = -self.calculate_angle_cost(angle_change, distance)
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 1, 1]

        elif action == 5:  # j turn left
            angle_change, distance = draw_angle_dist(4)
            reward = -self.calculate_angle_cost(angle_change, distance)
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 1, 1]

        elif action == 6:  # Do nothing
            self.prev_action_impulse = 0
            reward = -self.env_variables['rest_cost']

        elif action == 7:  # c start right
            angle_change, distance = draw_angle_dist(5)
            reward = -self.calculate_angle_cost(angle_change, distance)
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 0]

        elif action == 8:  # c start left
            angle_change, distance = draw_angle_dist(5)
            reward = -self.calculate_angle_cost(angle_change, distance)
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 0]

        elif action == 9:  # Approach swim.
            angle_change, distance = draw_angle_dist(10)
            reward = -self.calculate_angle_cost(angle_change, distance)
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        else:
            reward = None
            print("Invalid action given")

        return reward

    def try_impulse(self, impulse):
        # Used to produce calibration curve.
        self.body.apply_impulse_at_local_point((impulse, 0))
        return -self.env_variables['j_turn_cost']

    def readings_to_photons(self, readings):
        photons = np.random.poisson(readings * self.env_variables['photon_ratio'])
        if self.env_variables['read_noise_sigma'] > 0:
            noise = np.random.randn(readings.shape[0], readings.shape[1]) * self.env_variables['read_noise_sigma']
            photons += noise.astype(int)
        return photons

    def get_visual_inputs(self):
        left_photons = self.readings_to_photons(self.left_eye.readings)
        right_photons = self.readings_to_photons(self.right_eye.readings)
        left_eye = resize(np.reshape(left_photons, (1, len(self.left_eye.vis_angles), 3)) * (255 / self.env_variables['photon_ratio']), (20, self.env_variables['width'] / 2 - 50))
        right_eye = resize(np.reshape(right_photons, (1, len(self.right_eye.vis_angles), 3)) * (255 / self.env_variables['photon_ratio']), (20, self.env_variables['width'] / 2 - 50))
        eyes = np.hstack((left_eye, np.zeros((20, 100, 3)), right_eye))
        eyes[eyes < 0] = 0
        eyes[eyes > 255] = 255
        return eyes
