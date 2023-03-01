import numpy as np
import pymunk
from skimage.transform import resize, rescale

from Environment.Fish.eye import Eye
# from Environment.Action_Space.draw_angle_dist import draw_angle_dist
from Environment.Action_Space.draw_angle_dist_new import draw_angle_dist_new as draw_angle_dist


class Fish:
    """
    Created to simplify the SimState class, while making it easier to have environments with multiple agents in future.
    """

    def __init__(self, board, env_variables, dark_col, using_gpu, fish_mass=None):

        # For the purpose of producing a calibration curve.
        if fish_mass is None:
            inertia = pymunk.moment_for_circle(env_variables['fish_mass'], 0, env_variables['fish_head_size'], (0, 0))
        else:
            inertia = pymunk.moment_for_circle(fish_mass, 0, env_variables['fish_mouth_size'], (0, 0))

        self.env_variables = env_variables
        self.body = pymunk.Body(1, inertia)

        # Mouth
        self.mouth = pymunk.Circle(self.body, env_variables['fish_mouth_size'], offset=(0, 0))
        self.mouth.color = (0, 1, 0)
        self.mouth.elasticity = 1.0
        self.mouth.collision_type = 3

        # Head
        self.head = pymunk.Circle(self.body, env_variables['fish_head_size'],
                                  offset=(-env_variables['fish_head_size'], 0))
        self.head.color = (0, 1, 0)
        self.head.elasticity = 1.0
        self.head.collision_type = 6

        # # Tail
        tail_coordinates = ((-env_variables['fish_head_size'], 0),
                            (-env_variables['fish_head_size'], - env_variables['fish_head_size']),
                            (-env_variables['fish_head_size'] - env_variables['fish_tail_length'], 0),
                            (-env_variables['fish_head_size'], env_variables['fish_head_size']))
        self.tail = pymunk.Poly(self.body, tail_coordinates)
        self.tail.color = (0, 1, 0)
        self.tail.elasticity = 1.0
        self.tail.collision_type = 6

        # Init visual fields.
        self.verg_angle = env_variables['eyes_verg_angle'] * (np.pi / 180)
        self.retinal_field = env_variables['visual_field'] * (np.pi / 180)
        self.conv_state = 0

        max_visual_range = np.absolute(np.log(0.001) / self.env_variables["decay_rate"])

        self.left_eye = Eye(board, self.verg_angle, self.retinal_field, True, env_variables, dark_col, using_gpu,
                            max_visual_range=max_visual_range)
        self.right_eye = Eye(board, self.verg_angle, self.retinal_field, False, env_variables, dark_col, using_gpu,
                             max_visual_range=max_visual_range)

        self.hungry = 0
        self.stress = 1
        self.prey_consumed = False
        self.touched_edge = False
        self.touched_predator = False
        self.making_capture = False
        self.capture_possible = False
        self.prev_action_impulse = 0
        self.prev_action_angle = 0
        self.using_gpu = using_gpu

        # Energy system (new simulation)
        self.energy_level = 1.0
        self.ci = self.env_variables['ci']
        self.ca = self.env_variables['ca']
        self.baseline_decrease = self.env_variables['baseline_decrease']

        self.action_reward_scaling = self.env_variables['action_reward_scaling']
        self.consumption_reward_scaling = self.env_variables['consumption_reward_scaling']

        if "action_energy_use_scaling" in self.env_variables:
            self.action_energy_use_scaling = self.env_variables["action_energy_use_scaling"]
        else:
            self.action_energy_use_scaling = "Sublinear"

        # Salt health (new simulation)
        self.salt_health = 1.0

        # Touch edge - for penalty
        self.touched_edge_this_step = False

        self.impulse_vector_x = 0
        self.impulse_vector_y = 0

        if using_gpu:
            import cupy as cp

            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

    def take_action(self, action):
        """For discrete fish, overrided by continuous fish class."""
        if action == 0:  # Slow2
            angle_change, distance = draw_angle_dist(8)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = angle_change
            # self.prev_action_angle = np.random.choice([-angle_change, angle_change])
            self.body.angle += self.prev_action_angle
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 1:  # RT right
            angle_change, distance = draw_angle_dist(7)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = angle_change
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 2:  # RT left
            angle_change, distance = draw_angle_dist(7)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = -angle_change
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 3:  # Short capture swim
            angle_change, distance = draw_angle_dist(0)
            reward = -self.calculate_action_cost(angle_change, distance) - self.env_variables['capture_swim_extra_cost']
            self.prev_action_angle = angle_change
            # self.prev_action_angle = np.random.choice([-angle_change, angle_change])
            self.body.angle += self.prev_action_angle
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 1]
            self.making_capture = True

        elif action == 4:  # j turn right
            angle_change, distance = draw_angle_dist(4)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = angle_change
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 1, 1]

        elif action == 5:  # j turn left
            angle_change, distance = draw_angle_dist(4)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = -angle_change
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 1, 1]

        elif action == 6:  # Do nothing
            reward = 0
            self.prev_action_impulse = 0
            self.prev_action_angle = 0

        elif action == 7:  # c start right
            angle_change, distance = draw_angle_dist(5)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = angle_change
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 0]

        elif action == 8:  # c start left
            angle_change, distance = draw_angle_dist(5)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = -angle_change
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 0]

        elif action == 9:  # Approach swim.
            angle_change, distance = draw_angle_dist(10)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = angle_change
            # self.prev_action_angle = np.random.choice([-angle_change, angle_change])
            self.body.angle += self.prev_action_angle
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 10:  # j turn 1 right
            angle_change, distance = draw_angle_dist(44)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = angle_change
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 1, 1]

        elif action == 11:  # j turn 2 left
            angle_change, distance = draw_angle_dist(44)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.prev_action_angle = -angle_change
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 1, 1]

        else:
            reward = None
            print("Invalid action given")

        return reward

    def get_action_colour(self, action, magnitude, base_light):
        """Returns the (R, G, B) for associated actions"""
        if action == 0:  # Slow2
            action_colour = (base_light, magnitude, base_light)

        elif action == 1:  # RT right
            action_colour = (base_light, magnitude, base_light)

        elif action == 2:  # RT left
            action_colour = (base_light, magnitude, base_light)

        elif action == 3:  # Short capture swim
            action_colour = (magnitude, base_light, magnitude)

        elif action == 4:  # j turn right
            action_colour = (magnitude, magnitude, magnitude)

        elif action == 5:  # j turn left
            action_colour = (magnitude, magnitude, magnitude)

        elif action == 6:  # Do nothing
            action_colour = (0, 0, 0)

        elif action == 7:  # c start right
            action_colour = (magnitude, base_light, base_light)

        elif action == 8:  # c start left
            action_colour = (magnitude, base_light, base_light)

        elif action == 9:  # Approach swim.
            action_colour = (base_light, magnitude, base_light)

        elif action == 10:
            action_colour = (magnitude, magnitude, magnitude)

        elif action == 11:
            action_colour = (magnitude, magnitude, magnitude)

        else:
            action_colour = (0, 0, 0)
            print("Invalid action given")

        return action_colour

    def calculate_impulse(self, distance):
        """
        Uses the derived distance-mass-impulse relationship to convert an input distance (in mm) to impulse
        (arbitrary units).
        :param distance:
        :return:
        """
        # return (distance * 10 - (0.004644 * self.env_variables['fish_mass'] + 0.081417)) / 1.771548
        # return (distance * 10) * 0.360574383  # From mm
        return (distance * 10) * 0.34452532909386484  # From mm

    def calculate_action_cost(self, angle, distance):
        """
        So far, a fairly arbitrary equation to calculate action cost from distance moved and angle changed.
        cost = 0.05(angle change) + 1.5(distance moved)
        :return:
        """
        return abs(angle) * self.env_variables['angle_penalty_scaling_factor'] + (distance ** 2) * self.env_variables[
            'distance_penalty_scaling_factor']

    def try_impulse(self, impulse):
        # Used to produce calibration curve.
        self.body.apply_impulse_at_local_point((impulse, 0))
        return -self.env_variables['j_turn_cost']

    def readings_to_photons(self, readings):
        """Rounds down observations to form array of discrete photon events."""
        photons = np.floor(readings).astype(int)
        photons = photons.clip(0, 255)

        return photons

    def update_energy_level(self, reward, consumption):
        """Updates the current energy state for continuous and discrete fish."""
        energy_intake = 1.0 * consumption

        if self.action_energy_use_scaling == "Nonlinear":
            energy_use = self.ci * (abs(self.prev_action_impulse) ** 2) + self.ca * (
                        abs(self.prev_action_angle) ** 2) + self.baseline_decrease
        elif self.action_energy_use_scaling == "Linear":
            energy_use = self.ci * (abs(self.prev_action_impulse)) + self.ca * (
                abs(self.prev_action_angle)) + self.baseline_decrease
        elif self.action_energy_use_scaling == "Sublinear":
            energy_use = self.ci * (abs(self.prev_action_impulse) ** 0.5) + self.ca * (
                        abs(self.prev_action_angle) ** 0.5) + self.baseline_decrease
        else:
            energy_use = self.ci * (abs(self.prev_action_impulse) ** 0.5) + self.ca * (
                        abs(self.prev_action_angle) ** 0.5) + self.baseline_decrease

        reward += (energy_intake * self.consumption_reward_scaling) - (energy_use * self.action_reward_scaling)

        self.energy_level += energy_intake - energy_use
        if self.energy_level > 1.0:
            self.energy_level = 1.0

        return reward
