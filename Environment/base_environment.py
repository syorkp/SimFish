import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale
import pymunk

from Environment.Board.drawing_board import DrawingBoard


class BaseEnvironment:
    """A base class to represent environments, for extension to ProjectionEnvironment, VVR and Naturalistic
    environment classes."""

    def __init__(self, env_variables, using_gpu, num_actions):
        self.num_actions = num_actions

        self.env_variables = env_variables
        # Rescale bkg_scatter to avoid disruption for larger fields
        model = np.poly1d([1.32283913e-18, -4.10522256e-14, 4.92470049e-10, -2.86744090e-06, 8.22376164e-03,
                            4.07923942e-01])  # TODO: Keep parameters updated
        print(f"Original bkg scatter: {self.env_variables['bkg_scatter']}")
        self.env_variables["bkg_scatter"] = self.env_variables["bkg_scatter"] / (
                    model(self.env_variables["width"]) / model(1500))
        print(f"New bkg scatter: {self.env_variables['bkg_scatter']}")

        max_photoreceptor_rf_size = max([self.env_variables['uv_photoreceptor_rf_size'],
                                            self.env_variables['red_photoreceptor_rf_size']])
        if "light_gradient" in self.env_variables:
            light_gradient = self.env_variables['light_gradient']
        else:
            light_gradient = 0

        # Set max visual distance to the point at which 99.9% of photons have been lost to absorption mask.
        max_visual_distance = np.absolute(np.log(0.001)/self.env_variables["decay_rate"])

        self.board = DrawingBoard(arena_width=self.env_variables['width'],
                                  arena_height=self.env_variables['height'],
                                  uv_decay_rate=self.env_variables['decay_rate'],
                                  red_decay_rate=self.env_variables['decay_rate'],
                                  photoreceptor_rf_size=max_photoreceptor_rf_size,
                                  using_gpu=using_gpu,
                                  prey_size=self.env_variables['prey_size'],
                                  predator_size=self.env_variables['predator_size'],
                                  visible_scatter=self.env_variables['bkg_scatter'],
                                  dark_light_ratio=self.env_variables['dark_light_ratio'],
                                  dark_gain=self.env_variables['dark_gain'],
                                  light_gain=self.env_variables['light_gain'],
                                  light_gradient=light_gradient,
                                  max_visual_distance=max_visual_distance
                                  )

        self.show_all = False
        self.num_steps = 0
        self.fish = None

        self.dark_col = int(self.env_variables['width'] * self.env_variables['dark_light_ratio'])
        if self.dark_col == 0:  # Fixes bug with left wall always being invisible.
            self.dark_col = -1

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = self.env_variables['drag']

        self.prey_bodies = []
        self.prey_shapes = []

        self.prey_cloud_wall_shapes = []
        if self.env_variables["differential_prey"]:
            self.prey_cloud_locations = [
                [np.random.randint(
                    low=(self.env_variables['birth_rate_region_size'] / 2) + self.env_variables['prey_size'] +
                        self.env_variables['fish_mouth_size'],
                    high=self.env_variables['width'] - ((
                                                                self.env_variables['prey_size'] + self.env_variables[
                                                            'fish_mouth_size']) + (
                                                                self.env_variables['birth_rate_region_size'] / 2))),
                    np.random.randint(
                        low=(self.env_variables['birth_rate_region_size'] / 2) + self.env_variables['prey_size'] +
                            self.env_variables['fish_mouth_size'],
                        high=self.env_variables['height'] - ((
                                                                     self.env_variables['prey_size'] +
                                                                     self.env_variables[
                                                                         'fish_mouth_size']) + (self.env_variables[
                                                                                                    'birth_rate_region_size'] / 2)))]
                for cloud in range(int(self.env_variables["prey_cloud_num"]))]

            self.sand_grain_cloud_locations = [
                [np.random.randint(
                    low=120 + self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'],
                    high=self.env_variables['width'] - (
                            self.env_variables['sand_grain_size'] + self.env_variables[
                        'fish_mouth_size']) - 120),
                    np.random.randint(
                        low=120 + self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'],
                        high=self.env_variables['height'] - (
                                self.env_variables['sand_grain_size'] + self.env_variables[
                            'fish_mouth_size']) - 120)]
                for cloud in range(int(self.env_variables["sand_grain_num"]))]

        self.predator_bodies = []
        self.predator_shapes = []

        self.predator_body = None
        self.predator_shape = None
        self.predator_target = None

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []

        self.last_action = None

        self.prey_consumed_this_step = False

        self.predator_attacks_avoided = 0
        self.prey_caught = 0
        self.sand_grains_bumped = 0

        self.stimuli_information = {}

        # New energy system (log)
        self.energy_level_log = []

        # New complex predators
        self.predator_location = None
        self.remaining_predator_attacks = None
        self.total_predator_steps = None
        self.new_attack_due = False

        # New prey movement
        self.paramecia_gaits = None

        # For debugging purposes
        self.mask_buffer = []
        self.using_gpu = using_gpu

        if self.env_variables["salt"]:
            self.salt_gradient = None
            self.xp, self.yp = np.arange(self.env_variables['width']), np.arange(self.env_variables['height'])
            self.salt_damage_history = []
            self.salt_location = None

        if self.env_variables["prey_reproduction_mode"]:
            self.prey_ages = []

        # For initial loom attack
        self.first_attack = False
        self.loom_predator_current_size = None

        self.touched_sand_grain = False

        # For visualisation of previous actions
        self.action_buffer = []
        self.position_buffer = []
        self.fish_angle_buffer = []

        # For logging
        self.failed_capture_attempts = 0
        self.in_light_history = []
        self.survived_attack = False

        self.switch_step = None

    def clear_environmental_features(self):
        """Removes all prey, predators, and sand grains from simulation"""
        for i, shp in enumerate(self.prey_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.sand_grain_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.prey_cloud_wall_shapes):
            self.space.remove(shp)

        self.prey_cloud_wall_shapes = []

        self.paramecia_gaits = []

        if self.env_variables["prey_reproduction_mode"]:
            self.prey_ages = []


        if self.predator_shape is not None:
            self.remove_realistic_predator()
        self.predator_location = None
        self.remaining_predator_attacks = None
        self.total_predator_steps = None
        self.new_attack_due = False
        self.first_attack = False
        self.loom_predator_current_size = None

        self.prey_shapes = []
        self.prey_bodies = []

        self.predator_shapes = []
        self.predator_bodies = []

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []

    def draw_walls_and_sediment(self):
        self.board.erase(bkg=self.env_variables['bkg_scatter'])
        FOV = self.board.get_field_of_view(self.fish.body.position)
        self.board.draw_walls(FOV)
        self.board.draw_sediment(FOV)

    def draw_uv_shapes(self):
        #TODO: remove items outside visual range
        prey_pos = np.zeros((len(self.prey_bodies), 2), dtype=int)
        prey_pos[:, 0] = np.round(np.array([pr.position[0] for pr in self.prey_bodies]) - self.fish.body.position[0]).astype(int)
        prey_pos[:, 1] = np.round(np.array([pr.position[1] for pr in self.prey_bodies]) - self.fish.body.position[1]).astype(int)

        sand_pos = np.zeros((len(self.sand_grain_bodies), 2), dtype=int)
        sand_pos[:, 0] = np.round(np.array([sg.position[0] for sg in self.sand_grain_bodies]) - self.fish.body.position[0]).astype(int)
        sand_pos[:, 1] = np.round(np.array([sg.position[1] for sg in self.sand_grain_bodies]) - self.fish.body.position[1]).astype(int)

        self.board.draw_shapes_environmental(False, prey_pos, sand_pos)

    def reset(self):
        self.num_steps = 0
        self.fish.stress = 1
        self.fish.touched_edge_this_step = False
        self.prey_caught = 0
        self.predator_attacks_avoided = 0
        self.sand_grains_bumped = 0
        self.energy_level_log = []
        self.board.light_gain = self.env_variables["light_gain"]
        # No need for this to be computed each time? When config changes then creates new NaturalisticEnv and
        # DrawingBoard instances.
        # self.board.global_luminance_mask = self.board.get_luminance_mask(self.env_variables["dark_light_ratio"],
        #                                                                  self.env_variables["dark_gain"])

        self.switch_step = None

        # New energy system:
        self.fish.energy_level = 1

        # Reset salt gradient
        if self.env_variables["salt"]:
            self.reset_salt_gradient()
            self.fish.salt_health = 1.0
            self.salt_damage_history = []

        self.clear_environmental_features()
        self.board.reset()

        self.mask_buffer = []
        self.action_buffer = []
        self.position_buffer = []
        self.fish_angle_buffer = []

        self.failed_capture_attempts = 0
        self.in_light_history = []

    def reproduce_prey(self):
        num_prey = len(self.prey_bodies)
        # p_prey_birth = self.env_variables["birth_rate"] / (
        #         num_prey * self.env_variables["birth_rate_current_pop_scaling"])
        p_prey_birth = self.env_variables["birth_rate"] * (self.env_variables["prey_num"] - num_prey)
        for cloud in self.prey_cloud_locations:
            if np.random.rand(1) < p_prey_birth:
                if not self.check_proximity(cloud, self.env_variables["birth_rate_region_size"]):
                    new_location = (
                        np.random.randint(low=cloud[0] - (self.env_variables["birth_rate_region_size"] / 2),
                                          high=cloud[0] + (self.env_variables["birth_rate_region_size"] / 2)),
                        np.random.randint(low=cloud[1] - (self.env_variables["birth_rate_region_size"] / 2),
                                          high=cloud[1] + (self.env_variables["birth_rate_region_size"] / 2))
                    )
                    self.create_prey(new_location)
                    self.available_prey += 1

    def reset_salt_gradient(self, salt_source=None):
        if salt_source is None:
            salt_source_x = np.random.randint(0, self.env_variables['width'] - 1)
            salt_source_y = np.random.randint(0, self.env_variables['height'] - 1)
        else:
            salt_source_x = salt_source[0]
            salt_source_y = salt_source[1]

        self.salt_location = [salt_source_x, salt_source_y]
        salt_distance = (((salt_source_x - self.xp[:, None]) ** 2 + (
                salt_source_y - self.yp[None, :]) ** 2) ** 0.5)  # Measure of distance from source at every point.
        self.salt_gradient = np.exp(-self.env_variables["salt_concentration_decay"] * salt_distance) * \
                             self.env_variables["max_salt_damage"]

    def get_action_space_usage_display(self, current_shape):
        current_height = current_shape[0]
        current_width = current_shape[1]
        if self.continuous_actions:
            return self._get_action_space_usage_display_continuous(current_height, current_width)
        else:
            return self._get_action_space_usage_display_discrete(current_height, current_width)

    def _get_action_space_usage_display_continuous(self, current_height, current_width):
        difference = current_height - current_width
        extra_area = np.zeros((current_height, difference - 20, 3))
        available_height = current_width - 100

        impulse_resolution = 5
        angle_resolution = 25

        # Create counts for binned actions
        impulse_bins = np.linspace(0, self.env_variables["max_impulse"], int(self.env_variables["max_impulse"] * impulse_resolution))
        binned_impulses = np.digitize(np.array(self.action_buffer)[:, 0], impulse_bins)
        impulse_bin_counts = np.array([np.count_nonzero(binned_impulses == i) for i in range(len(impulse_bins))]).astype(float)

        angle_bins = np.linspace(-self.env_variables["max_angle_change"], self.env_variables["max_angle_change"],
                                 int(self.env_variables["max_angle_change"] * angle_resolution))
        binned_angles = np.digitize(np.array(self.action_buffer)[:, 1], angle_bins)
        angle_bin_counts = np.array([np.count_nonzero(binned_angles == i) for i in range(len(angle_bins))]).astype(float)

        impulse_bin_scaling = (difference-20)/max(impulse_bin_counts)
        angle_bin_scaling = (difference-20)/max(angle_bin_counts)

        impulse_bin_counts *= impulse_bin_scaling
        angle_bin_counts *= angle_bin_scaling

        impulse_bin_counts = np.floor(impulse_bin_counts).astype(int)
        angle_bin_counts = np.floor(angle_bin_counts).astype(int)

        bin_height = int(math.floor(available_height / (len(impulse_bin_counts) + len(angle_bin_counts))))

        current_h = 0
        for count in impulse_bin_counts:
            extra_area[current_h:current_h+bin_height, 0:count, :] = 255.0
            current_h += bin_height

        current_h += 100

        for count in angle_bin_counts:
            extra_area[current_h:current_h+bin_height, 0:count, :] = 255.0
            current_h += bin_height

        return extra_area

    def _get_action_space_usage_display_discrete(self, current_height, current_width):
        difference = current_height - current_width
        extra_area = np.zeros((current_height, difference - 20, 3))

        action_bins = [i for i in range(self.num_actions)]
        # binned_actions = np.digitize(np.array(self.action_buffer), action_bins)
        action_bin_counts = np.array([np.count_nonzero(np.array(self.action_buffer) == i) for i in action_bins]).astype(float)

        action_bin_scaling = (difference-20)/max(action_bin_counts)
        action_bin_counts *= action_bin_scaling
        action_bin_counts = np.floor(action_bin_counts).astype(int)

        bin_height = int(math.floor(current_width/len(action_bins)))

        current_h = 0

        for count in action_bin_counts:
            extra_area[current_h:current_h+bin_height, 0:count, :] = 255.0
            current_h += bin_height

        return extra_area

    def build_prey_cloud_walls(self):
        for i in self.prey_cloud_locations:
            wall_edges = [
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] - 150), (i[0] - 150, i[1] + 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] + 150), (i[0] + 150, i[1] + 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] + 150, i[1] + 150), (i[0] + 150, i[1] - 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] - 150), (i[0] + 150, i[1] - 150), 1)
            ]
            for s in wall_edges:
                s.friction = 1.
                s.group = 1
                s.collision_type = 7
                s.color = (0, 0, 0)
                self.space.add(s)
                self.prey_cloud_wall_shapes.append(s)

    def create_walls(self):
        # wall_width = 1
        wall_width = 5  # self.env_variables['eyes_biasx']
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, wall_width), (0, self.env_variables['height']), wall_width),
            pymunk.Segment(
                self.space.static_body,
                (wall_width, self.env_variables['height']), (self.env_variables['width'], self.env_variables['height']),
                wall_width),
            pymunk.Segment(
                self.space.static_body,
                (self.env_variables['width'] - wall_width, self.env_variables['height']),
                (self.env_variables['width'] - wall_width, wall_width),
                wall_width),
            pymunk.Segment(
                self.space.static_body,
                (wall_width, wall_width), (self.env_variables['width'], wall_width), wall_width)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = (1, 0, 0)
            self.space.add(s)

        # self.space.add(static)

    @staticmethod
    def no_collision(arbiter, space, data):
        return False

    def touch_wall(self, arbiter, space, data):
        if not self.env_variables["wall_reflection"]:
            return self._touch_wall(arbiter, space, data)
        else:
            return self._touch_wall_reflect(arbiter, space, data)

    def _touch_wall_reflect(self, arbiter, space, data):
        # print(f"Fish touched wall: {self.fish.body.position}")
        new_position_x = self.fish.body.position[0]
        new_position_y = self.fish.body.position[1]

        if new_position_x < 40:  # Wall d
            new_position_x = 40 + self.env_variables["fish_head_size"] + \
                             self.env_variables["fish_tail_length"]
        elif new_position_x > self.env_variables['width'] - 40:  # wall b
            new_position_x = self.env_variables['width'] - (
                    40 + self.env_variables["fish_head_size"] +
                    self.env_variables["fish_tail_length"])
        if new_position_y < 40:  # wall a
            new_position_y = 40 + self.env_variables["fish_head_size"] + \
                             self.env_variables["fish_tail_length"]
        elif new_position_y > self.env_variables['height'] - 40:  # wall c
            new_position_y = self.env_variables['height'] - (
                    40 + self.env_variables["fish_head_size"] +
                    self.env_variables["fish_tail_length"])

        new_position = pymunk.Vec2d(new_position_x, new_position_y)
        self.fish.body.position = new_position
        self.fish.body.velocity = (0, 0)

        if self.fish.body.angle < np.pi:
            self.fish.body.angle += np.pi
        else:
            self.fish.body.angle -= np.pi
        self.fish.touched_edge = True
        return True

    def _touch_wall(self, arbiter, space, data):
        position_x = self.fish.body.position[0]
        position_y = self.fish.body.position[1]

        if position_x < 8:
            new_position_x = 10
        elif position_x > self.env_variables["width"] - 7:
            new_position_x = self.env_variables["width"] - 9

        if position_y < 8:
            new_position_y = 10
        elif position_y > self.env_variables["height"] - 7:
            new_position_y = self.env_variables["height"] - 9

        if "new_position_x" in locals():
            new_position = pymunk.Vec2d(new_position_x, self.fish.body.position[1])
            self.fish.body.position = new_position
            self.fish.body.velocity = (0, 0)

        if "new_position_y" in locals():
            new_position = pymunk.Vec2d(self.fish.body.position[0], new_position_y)
            self.fish.body.position = new_position
            self.fish.body.velocity = (0, 0)

        self.fish.touched_edge = True
        self.fish.touched_edge_this_step = True
        return True

    def create_prey(self, prey_position=None, prey_orientation=None):
        self.prey_bodies.append(pymunk.Body(self.env_variables['prey_mass'], self.env_variables['prey_inertia']))
        self.prey_shapes.append(pymunk.Circle(self.prey_bodies[-1], self.env_variables['prey_size']))
        self.prey_shapes[-1].elasticity = 1.0
        self.prey_bodies[-1].angle = np.random.uniform(0, np.pi*2)
        if prey_position is None:
            if not self.env_variables["differential_prey"]:
                self.prey_bodies[-1].position = (
                    np.random.randint(
                        self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'] + 40,
                        self.env_variables['width'] - (
                                self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'] +
                                40)),
                    np.random.randint(
                        self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'] + 40,
                        self.env_variables['height'] - (
                                self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'] +
                                40)))
            else:
                cloud = random.choice(self.prey_cloud_locations)
                self.prey_bodies[-1].position = (
                    np.random.randint(low=cloud[0] - (self.env_variables["birth_rate_region_size"] / 2),
                                      high=cloud[0] + (self.env_variables["birth_rate_region_size"] / 2)),
                    np.random.randint(low=cloud[1] - (self.env_variables["birth_rate_region_size"] / 2),
                                      high=cloud[1] + (self.env_variables["birth_rate_region_size"] / 2))
                )
        else:
            self.prey_bodies[-1].position = prey_position

        self.prey_shapes[-1].color = (0, 0, 1)
        self.prey_shapes[-1].collision_type = 2
        # self.prey_shapes[-1].filter = pymunk.ShapeFilter(
        #     mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.prey_bodies[-1], self.prey_shapes[-1])

        # New prey motion
        self.paramecia_gaits.append(
            np.random.choice([0, 1, 2], 1, p=[1 - (self.env_variables["p_fast"] + self.env_variables["p_slow"]),
                                              self.env_variables["p_slow"],
                                              self.env_variables["p_fast"]])[0])

        if self.env_variables["prey_reproduction_mode"]:
            self.prey_ages.append(0)

    def check_proximity(self, feature_position, sensing_distance):
        sensing_area = [[feature_position[0] - sensing_distance,
                         feature_position[0] + sensing_distance],
                        [feature_position[1] - sensing_distance,
                         feature_position[1] + sensing_distance]]
        is_in_area = sensing_area[0][0] <= self.fish.body.position[0] <= sensing_area[0][1] and \
                     sensing_area[1][0] <= self.fish.body.position[1] <= sensing_area[1][1]
        if is_in_area:
            return True
        else:
            return False

    def check_proximity_all_prey(self, sensing_distance):
        # all_prey_positions = np.array([pr.position for pr in self.prey_bodies])
        # fish_position = self.fish.body.position
        # within_x = (all_prey_positions[:, 0] > fish_position[0] - sensing_distance) * (all_prey_positions[:, 0] < fish_position[0] + sensing_distance)
        # within_y = (all_prey_positions[:, 1] > fish_position[1] - sensing_distance) * (all_prey_positions[:, 1] < fish_position[1] + sensing_distance)
        # within_range = within_x * within_y

        all_prey_positions = np.array([pr.position for pr in self.prey_bodies])
        fish_position = np.expand_dims(np.array(self.fish.body.position), 0)
        fish_prey_vectors = all_prey_positions - fish_position

        fish_prey_distances = ((fish_prey_vectors[:, 0] ** 2) + (fish_prey_vectors[:, 1] ** 2) ** 0.5)
        within_range = fish_prey_distances < sensing_distance
        return within_range

    def get_fish_prey_incidence(self):
        fish_orientation = self.fish.body.angle
        fish_position = np.expand_dims(np.array(self.fish.body.position), axis=0)
        paramecium_positions = np.array([pr.position for pr in self.prey_bodies])
        fish_orientation = np.array([fish_orientation])

        fish_orientation_sign = ((fish_orientation >= 0) * 1) + ((fish_orientation < 0) * -1)

        # Remove full orientations (so is between -2pi and 2pi
        fish_orientation %= 2 * np.pi * fish_orientation_sign

        # Convert to positive scale between 0 and 2pi
        fish_orientation[fish_orientation < 0] += 2 * np.pi

        fish_prey_vectors = paramecium_positions - fish_position

        # Adjust according to quadrents.
        fish_prey_angles = np.arctan(fish_prey_vectors[:, 1] / fish_prey_vectors[:, 0])

        #   Generates positive angle from left x axis clockwise.
        # UL quadrent
        in_ul_quadrent = (fish_prey_vectors[:, 0] < 0) * (fish_prey_vectors[:, 1] > 0)
        fish_prey_angles[in_ul_quadrent] += np.pi
        # BR quadrent
        in_br_quadrent = (fish_prey_vectors[:, 0] > 0) * (fish_prey_vectors[:, 1] < 0)
        fish_prey_angles[in_br_quadrent] += (np.pi * 2)
        # BL quadrent
        in_bl_quadrent = (fish_prey_vectors[:, 0] < 0) * (fish_prey_vectors[:, 1] < 0)
        fish_prey_angles[in_bl_quadrent] += np.pi

        # Angle ends up being between 0 and 2pi as clockwise from right x-axis. Same frame as fish angle:
        fish_prey_incidence = np.expand_dims(np.array([fish_orientation]), 1) - fish_prey_angles

        fish_prey_incidence[fish_prey_incidence > np.pi] %= np.pi
        fish_prey_incidence[fish_prey_incidence < -np.pi] %= -np.pi

        return fish_prey_incidence

    def move_prey(self, micro_step):
        if len(self.prey_bodies) == 0:
            return

        # Generate impulses
        impulse_types = [0, self.env_variables["slow_speed_paramecia"], self.env_variables["fast_speed_paramecia"]]
        impulses = [impulse_types[gait] for gait in self.paramecia_gaits]

        # Do once per step.
        if micro_step == 0:
            gaits_to_switch = np.random.choice([0, 1], len(self.prey_shapes),
                                               p=[1 - self.env_variables["p_switch"], self.env_variables["p_switch"]])
            switch_to = np.random.choice([0, 1, 2], len(self.prey_shapes),
                                         p=[1 - (self.env_variables["p_slow"] + self.env_variables["p_fast"]),
                                            self.env_variables["p_slow"], self.env_variables["p_fast"]])
            self.paramecia_gaits = [switch_to[i] if gaits_to_switch[i] else old_gait for i, old_gait in
                                    enumerate(self.paramecia_gaits)]

            # Angles of change
            angle_changes = np.random.uniform(-self.env_variables['prey_max_turning_angle'],
                                              self.env_variables['prey_max_turning_angle'],
                                              len(self.prey_shapes))

            # Large angle changes
            large_turns = np.random.uniform(-np.pi, np.pi, len(self.prey_shapes))
            turns_implemented = np.random.choice([0, 1], len(self.prey_shapes), p=[1-self.env_variables["p_reorient"],
                                                                                   self.env_variables["p_reorient"]])
            angle_changes = angle_changes + (large_turns * turns_implemented)

            self.prey_within_range = self.check_proximity_all_prey(self.env_variables["prey_sensing_distance"])

        for i, prey_body in enumerate(self.prey_bodies):
            if self.prey_within_range[i]:  # self.check_proximity(prey_body.position, self.env_variables['prey_sensing_distance']):
                # Motion from fluid dynamics
                if self.env_variables["prey_fluid_displacement"]:
                    distance_vector = prey_body.position - self.fish.body.position
                    distance = (distance_vector[0] ** 2 + distance_vector[1] ** 2) ** 0.5
                    distance_scaling = np.exp(-distance)

                    original_angle = copy.copy(prey_body.angle)
                    prey_body.angle = self.fish.body.angle + np.random.uniform(-1, 1)
                    impulse_for_prey = (self.get_last_action_magnitude()/self.env_variables["known_max_fish_i"]) * \
                                        self.env_variables["displacement_scaling_factor"] * distance_scaling

                    prey_body.apply_impulse_at_local_point((impulse_for_prey, 0))
                    prey_body.angle = original_angle

                # Motion from prey escape
                if self.env_variables["prey_jump"] and np.random.choice([0, 1], size=1,
                                                                        p=[1 - self.env_variables["p_escape"]/self.env_variables["phys_steps_per_sim_step"],
                                                                           self.env_variables["p_escape"]/self.env_variables["phys_steps_per_sim_step"]])[0] == 1:
                    prey_body.apply_impulse_at_local_point((self.env_variables["jump_speed_paramecia"], 0))

            else:
                if micro_step == 0:
                    prey_body.angle = prey_body.angle + angle_changes[i]

                prey_body.apply_impulse_at_local_point((impulses[i], 0))

    def touch_prey(self, arbiter, space, data):
        valid_capture = False
        if self.fish.capture_possible:
            for i, shp in enumerate(self.prey_shapes):
                if shp == arbiter.shapes[0]:
                    # Check if angles line up.
                    prey_position = self.prey_bodies[i].position
                    fish_position = self.fish.body.position
                    vector = prey_position - fish_position  # Taking fish as origin

                    # Will generate values between -pi/2 and pi/2 which require adjustment depending on quadrant.
                    angle = np.arctan(vector[1] / vector[0])

                    if vector[0] < 0 and vector[1] < 0:
                        # Generates postiive angle from left x axis clockwise.
                        # print("UL quadrent")
                        angle += np.pi
                    elif vector[1] < 0:
                        # Generates negative angle from right x axis anticlockwise.
                        # print("UR quadrent.")
                        angle = angle + (np.pi * 2)
                    elif vector[0] < 0:
                        # Generates negative angle from left x axis anticlockwise.
                        # print("BL quadrent.")
                        angle = angle + np.pi

                    # Angle ends up being between 0 and 2pi as clockwise from right x axis. Same frame as fish angle:
                    fish_orientation = (self.fish.body.angle % (2 * np.pi))

                    # Normalise so both in same reference frame
                    deviation = abs(fish_orientation - angle)
                    if deviation > np.pi:
                        # Need to account for cases where one angle is very high, while other is very low, as these
                        # angles can be close together. Can do this by summing angles and subtracting from 2 pi.

                        # deviation = abs((2*np.pi)-(fish_orientation+angle))
                        deviation -= (2 * np.pi)
                        deviation = abs(deviation)
                    if deviation < self.env_variables["capture_angle_deviation_allowance"]:
                        # print("Successful capture \n")
                        valid_capture = True
                        space.remove(shp, shp.body)
                        self.prey_shapes.remove(shp)
                        self.prey_bodies.remove(shp.body)
                        del self.paramecia_gaits[i]
                        if self.env_variables["prey_reproduction_mode"]:
                            del self.prey_ages[i]
                    else:
                        self.failed_capture_attempts += 1
                        # print("Failed capture \n")
                        # print(f"""Prey position: {prey_position}
                        # Fish position: {fish_position}
                        # Fish orientation: {fish_orientation}
                        # Computed orientation: {angle}
                        # """)

            if valid_capture:
                self.prey_caught += 1
                self.fish.prey_consumed = True
                self.prey_consumed_this_step = True

            return False
        else:
            self.failed_capture_attempts += 1
            return True

    def remove_prey(self, prey_index):
        self.space.remove(self.prey_shapes[prey_index], self.prey_shapes[prey_index].body)
        del self.prey_shapes[prey_index]
        del self.prey_bodies[prey_index]
        del self.prey_ages[prey_index]
        del self.paramecia_gaits[prey_index]

    def create_predator(self):
        self.predator_bodies.append(
            pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia']))
        self.predator_shapes.append(pymunk.Circle(self.predator_bodies[-1], self.env_variables['predator_size']))
        self.predator_shapes[-1].elasticity = 1.0
        self.predator_bodies[-1].position = (
            np.random.randint(self.env_variables['predator_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['width'] - (
                                      self.env_variables['predator_size'] + self.env_variables['fish_mouth_size'])),
            np.random.randint(self.env_variables['predator_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['height'] - (
                                      self.env_variables['predator_size'] + self.env_variables['fish_mouth_size'])))
        self.predator_shapes[-1].color = (0, 1, 0)
        # Made green so still visible to us but not to fish.
        self.predator_shapes[-1].collision_type = 5

        self.space.add(self.predator_bodies[-1], self.predator_shapes[-1])

    def move_predator(self, micro_step):
        # OLD:
        # for pr in self.predator_bodies:
        #     dist_to_fish = np.sqrt(
        #         (pr.position[0] - self.fish.body.position[0]) ** 2 + (pr.position[1] - self.fish.body.position[1]) ** 2)
        #
        #     if dist_to_fish < self.env_variables['predator_sensing_dist']:
        #         pr.angle = np.pi / 2 - np.arctan2(self.fish.body.position[0] - pr.position[0],
        #                                           self.fish.body.position[1] - pr.position[1])
        #         pr.apply_impulse_at_local_point((self.env_variables['predator_chase_impulse'], 0))
        #
        #     elif np.random.rand(1) < self.env_variables['predator_impulse_rate']:
        #         pr.angle = np.random.rand(1) * 2 * np.pi
        #         pr.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))
        if self.first_attack:
            # If the first attack is to be a loom attack (specified by selecting loom stimulus in env config)
            if self.loom_predator_current_size < self.env_variables["final_predator_size"]:
                if micro_step == 0:
                    self.grow_loom_predator()
            else:
                self.remaining_predator_attacks -= 1
                self.predator_attacks_avoided += 1
                self.new_attack_due = True
                self.first_attack = False

                # Remove the loom predator
                self.space.remove(self.predator_shape, self.predator_shape.body)

                # Create new predator
                self.predator_body = pymunk.Body(self.env_variables['predator_mass'],
                                                 self.env_variables['predator_inertia'])
                self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_size'])
                self.predator_shape.elasticity = 1.0
                self.predator_body.position = self.predator_location
                self.predator_shape.color = (0, 1, 0)
                self.predator_shape.collision_type = 5
                self.predator_shape.filter = pymunk.ShapeFilter(
                    mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # Category 2 objects cant collide with predator
                self.space.add(self.predator_body, self.predator_shape)
        else:
            if self.check_predator_at_target():
                self.remaining_predator_attacks -= 1
                self.predator_attacks_avoided += 1
                self.new_attack_due = True

            if self.check_predator_outside_walls():
                self.remaining_predator_attacks -= 1

            # If predator out of strike range.
            if self.predator_base_distance_to_fish() > self.env_variables["max_predator_attack_range"]:
                self.predator_body.position = self.predator_location
                return

            if self.remaining_predator_attacks <= 0 or \
                    self.total_predator_steps > self.env_variables["predator_presence_duration_steps"]:
                self.remove_realistic_predator()
                return
            else:
                if self.new_attack_due and self.check_fish_not_near_wall():
                    self.new_attack_due = False
                    self.initiate_repeated_predator_attack()

            # Update predator target
            # if self.predator_distance_to_fish() > self.env_variables["max_predator_reorient_distance"]:
            #     self.predator_target = np.array(self.fish.body.position)

            self.predator_body.angle = np.pi / 2 - np.arctan2(
                self.predator_target[0] - self.predator_body.position[0],
                self.predator_target[1] - self.predator_body.position[1])
            self.predator_body.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def touch_predator(self, arbiter, space, data):
        if self.num_steps > self.env_variables['immunity_steps']:
            self.fish.touched_predator = True
            return False
        else:
            return True

    def check_fish_proximity_to_walls(self):
        fish_position = self.fish.body.position

        # Check proximity to left wall
        if 0 < fish_position[0] < self.env_variables["distance_from_fish"]:
            left = True
        else:
            left = False

        # Check proximity to right wall
        if self.env_variables["width"] - self.env_variables["distance_from_fish"] < fish_position[0] < \
                self.env_variables["width"]:
            right = True
        else:
            right = False

        # Check proximity to bottom wall
        if self.env_variables["height"] - self.env_variables["distance_from_fish"] < fish_position[1] < \
                self.env_variables["height"]:
            bottom = True
        else:
            bottom = False

        # Check proximity to top wall
        if 0 < fish_position[0] < self.env_variables["distance_from_fish"]:
            top = True
        else:
            top = False

        return left, bottom, right, top

    def select_predator_angle_of_attack(self):
        left, bottom, right, top = self.check_fish_proximity_to_walls()
        if left and top:
            angle_from_fish = random.randint(90, 180)
        elif left and bottom:
            angle_from_fish = random.randint(0, 90)
        elif right and top:
            angle_from_fish = random.randint(180, 270)
        elif right and bottom:
            angle_from_fish = random.randint(270, 360)
        elif left:
            angle_from_fish = random.randint(0, 180)
        elif top:
            angle_from_fish = random.randint(90, 270)
        elif bottom:
            angles = [random.randint(270, 360), random.randint(0, 90)]
            angle_from_fish = random.choice(angles)
        elif right:
            angle_from_fish = random.randint(180, 360)
        else:
            angle_from_fish = random.randint(0, 360)

        angle_from_fish = np.radians(angle_from_fish / np.pi)
        return angle_from_fish

    def check_fish_not_near_wall(self):
        buffer_region = self.env_variables["predator_size"] * 1.5
        x_position, y_position = self.fish.body.position[0], self.fish.body.position[1]

        if x_position < buffer_region:
            return True
        elif x_position > self.env_variables["width"] - buffer_region:
            return True
        if y_position < buffer_region:
            return True
        elif y_position > self.env_variables["height"] - buffer_region:
            return True

    def create_realistic_predator_existing(self, predator_position, predator_orientation, predator_target):

        self.predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])
        self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_size'])
        self.predator_shape.elasticity = 1.0

        self.predator_body.position = (predator_position[0], predator_position[1])
        self.predator_body.angle = predator_orientation
        self.predator_target = predator_target
        self.total_predator_steps = 0

        self.predator_shape.color = (0, 1, 0)
        self.predator_location = (predator_position[0], predator_position[1])
        self.remaining_predator_attacks = 1 + np.sum(
            np.random.choice([0, 1], self.env_variables["max_predator_attacks"] - 1,
                                p=[1.0 - self.env_variables["further_attack_probability"],
                                self.env_variables["further_attack_probability"]]))
        if self.env_variables["predator_first_attack_loom"]:
            # Set fish position based on final predator size
            self.predator_location = (predator_position[0], predator_position[1])

            self.predator_body.position = self.predator_location
            self.loom_predator_current_size = self.env_variables['initial_predator_size']
            self.first_attack = True

        self.predator_shape.collision_type = 5
        self.predator_shape.filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # Category 2 objects cant collide with predator

        self.space.add(self.predator_body, self.predator_shape)


    def create_realistic_predator(self, predator_position=None, predator_orientation=None, predator_target=None):
        if predator_position is not None:
            return self.create_realistic_predator_existing(predator_position, predator_orientation, predator_target)

        self.predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])
        self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_size'])
        self.predator_shape.elasticity = 1.0

        fish_position = self.fish.body.position

        angle_from_fish = self.select_predator_angle_of_attack()
        dy = self.env_variables["distance_from_fish"] * np.cos(angle_from_fish)
        dx = self.env_variables["distance_from_fish"] * np.sin(angle_from_fish)

        x_position = fish_position[0] + dx
        y_position = fish_position[1] + dy

        self.predator_body.position = (x_position, y_position)
        self.predator_target = fish_position
        self.total_predator_steps = 0

        self.predator_shape.color = (0, 1, 0)
        self.predator_location = (x_position, y_position)
        self.remaining_predator_attacks = 1 + np.sum(
            np.random.choice([0, 1], self.env_variables["max_predator_attacks"] - 1,
                                p=[1.0 - self.env_variables["further_attack_probability"],
                                self.env_variables["further_attack_probability"]]))
        if self.env_variables["predator_first_attack_loom"]:
            # Set fish position based on final predator size
            dx = (self.env_variables["final_predator_size"] * 0.8) * np.sin(angle_from_fish)
            dy = (self.env_variables["final_predator_size"] * 0.8) * np.cos(angle_from_fish)
            self.predator_location = (fish_position[0] + dx, fish_position[1] + dy)

            self.predator_body.position = self.predator_location
            self.loom_predator_current_size = self.env_variables['initial_predator_size']
            self.first_attack = True

        self.predator_shape.collision_type = 5
        self.predator_shape.filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # Category 2 objects cant collide with predator

        self.space.add(self.predator_body, self.predator_shape)

    def grow_loom_predator(self):
        self.loom_predator_current_size += (self.env_variables["final_predator_size"] - self.env_variables[
            "initial_predator_size"]) / self.env_variables["duration_of_loom"]
        # Remove the predator as it appears
        self.space.remove(self.predator_shape, self.predator_shape.body)

        # Create new predator:
        self.predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])
        self.predator_shape = pymunk.Circle(self.predator_body, self.loom_predator_current_size)
        self.predator_shape.elasticity = 1.0
        self.predator_body.position = self.predator_location
        self.predator_shape.color = (0, 1, 0)
        self.predator_shape.collision_type = 5
        self.predator_shape.filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # Category 2 objects cant collide with predator
        self.space.add(self.predator_body, self.predator_shape)

    def check_predator_outside_walls(self):
        x_position, y_position = self.predator_body.position[0], self.predator_body.position[1]
        if x_position < 0:
            return True
        elif x_position > self.env_variables["width"]:
            return True
        if y_position < 0:
            return True
        elif y_position > self.env_variables["height"]:
            return True

    def check_predator_at_target(self):
        if (round(self.predator_body.position[0]), round(self.predator_body.position[1])) == (
                round(self.predator_target[0]), round(self.predator_target[1])):
            return True
        else:
            return False

    def predator_base_distance_to_fish(self):
        return ((self.predator_location[0] - self.fish.body.position[0]) ** 2 +
                (self.predator_location[1] - self.fish.body.position[1]) ** 2) ** 0.5

    def predator_distance_to_fish(self):
        return ((self.predator_body.position[0] - self.fish.body.position[0]) ** 2 +
                (self.predator_body.position[1] - self.fish.body.position[1]) ** 2) ** 0.5

    def move_realistic_predator(self, micro_step):
        if self.first_attack:
            # If the first attack is to be a loom attack (specified by selecting loom stimulus in env config)
            if self.loom_predator_current_size < self.env_variables["final_predator_size"]:
                if micro_step == 0:
                    self.grow_loom_predator()
            else:
                self.remaining_predator_attacks -= 1
                self.predator_attacks_avoided += 1
                self.new_attack_due = True
                self.first_attack = False

                # Remove the loom predator
                self.space.remove(self.predator_shape, self.predator_shape.body)

                # Create new predator
                self.predator_body = pymunk.Body(self.env_variables['predator_mass'],
                                                 self.env_variables['predator_inertia'])
                self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_size'])
                self.predator_shape.elasticity = 1.0
                self.predator_body.position = self.predator_location
                self.predator_shape.color = (0, 1, 0)
                self.predator_shape.collision_type = 5
                self.predator_shape.filter = pymunk.ShapeFilter(
                    mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # Category 2 objects cant collide with predator
                self.space.add(self.predator_body, self.predator_shape)
        else:
            if self.check_predator_at_target():
                self.remaining_predator_attacks -= 1
                self.predator_attacks_avoided += 1
                self.new_attack_due = True

            if self.check_predator_outside_walls():
                self.remaining_predator_attacks -= 1

            # If predator out of strike range.
            if self.predator_base_distance_to_fish() > self.env_variables["max_predator_attack_range"]:
                self.predator_body.position = self.predator_location
                return

            if self.remaining_predator_attacks <= 0 or \
                    self.total_predator_steps > self.env_variables["predator_presence_duration_steps"]:
                self.remove_realistic_predator()
                return
            else:
                if self.new_attack_due and self.check_fish_not_near_wall():
                    self.new_attack_due = False
                    self.initiate_repeated_predator_attack()

            # Update predator target
            # if self.predator_distance_to_fish() > self.env_variables["max_predator_reorient_distance"]:
            #     self.predator_target = np.array(self.fish.body.position)

            self.predator_body.angle = np.pi / 2 - np.arctan2(
                self.predator_target[0] - self.predator_body.position[0],
                self.predator_target[1] - self.predator_body.position[1])
            self.predator_body.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def remove_realistic_predator(self, arbiter=None, space=None, data=None):
        if self.predator_body is not None:
            self.space.remove(self.predator_shape, self.predator_shape.body)
            self.predator_shape = None
            self.predator_body = None
            self.predator_location = None
            self.predator_target = None
            self.remaining_predator_attacks = None
            self.total_predator_steps = None
            self.survived_attack = True
        else:
            pass

    def initiate_repeated_predator_attack(self):
        self.space.remove(self.predator_shape, self.predator_shape.body)
        self.predator_body.position = self.predator_location

        fish_position = self.fish.body.position
        self.predator_target = fish_position
        self.space.add(self.predator_body, self.predator_shape)

    def create_sand_grain(self):
        self.sand_grain_bodies.append(
            pymunk.Body(self.env_variables['sand_grain_mass'], self.env_variables['sand_grain_inertia']))
        self.sand_grain_shapes.append(pymunk.Circle(self.sand_grain_bodies[-1], self.env_variables['sand_grain_size']))
        self.sand_grain_shapes[-1].elasticity = 1.0

        if not self.env_variables["differential_prey"]:
            self.sand_grain_bodies[-1].position = (
                np.random.randint(self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'],
                                  self.env_variables['width'] - (
                                          self.env_variables['sand_grain_size'] + self.env_variables[
                                      'fish_mouth_size'])),
                np.random.randint(self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'],
                                  self.env_variables['height'] - (
                                          self.env_variables['sand_grain_size'] + self.env_variables[
                                      'fish_mouth_size'])))
        else:
            cloud = random.choice(self.sand_grain_cloud_locations)
            self.sand_grain_bodies[-1].position = (
                np.random.randint(low=cloud[0] - (self.env_variables["birth_rate_region_size"] / 2),
                                  high=cloud[0] + (self.env_variables["birth_rate_region_size"] / 2)),
                np.random.randint(low=cloud[1] - (self.env_variables["birth_rate_region_size"] / 2),
                                  high=cloud[1] + (self.env_variables["birth_rate_region_size"] / 2))
            )

        self.sand_grain_shapes[-1].color = (0, 0, 1)

        self.sand_grain_shapes[-1].collision_type = 4
        self.sand_grain_shapes[-1].filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.sand_grain_bodies[-1], self.sand_grain_shapes[-1])

    def touch_grain(self, arbiter, space, data):
        self.fish.touched_sand_grain = True

        if self.last_action == 3:
            self.sand_grains_bumped += 1

    def get_last_action_magnitude(self):
        return self.fish.prev_action_impulse * self.env_variables[
            'displacement_scaling_factor']  # Scaled down both for mass effects and to make it possible for the prey to be caught.

    def displace_sand_grains(self):
        for i, body in enumerate(self.sand_grain_bodies):
            if self.check_proximity(self.sand_grain_bodies[i].position,
                                    self.env_variables['sand_grain_displacement_distance']):
                self.sand_grain_bodies[i].angle = self.fish.body.angle + np.random.uniform(-1, 1)
                # if self.sand_grain_bodies[i].angle < (3 * np.pi) / 2:
                #     self.sand_grain_bodies[i].angle += np.pi / 2
                # else:
                #     self.sand_grain_bodies[i].angle -= np.pi / 2
                self.sand_grain_bodies[i].apply_impulse_at_local_point(
                    (self.get_last_action_magnitude(), 0))

                # if "sand_grain_touch_penalty" in self.env_variables:
                #     penalty -= self.env_variables["sand_grain_touch_penalty"]
