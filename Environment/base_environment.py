import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale
import pymunk

from Tools.drawing_board import DrawingBoard
from Tools.drawing_board_new import NewDrawingBoard


class BaseEnvironment:
    """A base class to represent environments, for extension to ProjectionEnvironment, VVR and Naturalistic
    environment classes."""

    def __init__(self, env_variables, draw_screen, new_simulation, using_gpu, num_actions):
        self.new_simulation = new_simulation
        self.num_actions = num_actions

        self.env_variables = env_variables
        if self.new_simulation:
            # Rescale bkg_scatter to avoid disruption for larger fields

            model = np.poly1d([1.32283913e-18, -4.10522256e-14, 4.92470049e-10, -2.86744090e-06, 8.22376164e-03,
                               4.07923942e-01])  # TODO: Keep parameters updated
            self.env_variables["bkg_scatter"] = self.env_variables["bkg_scatter"] / (
                        model(self.env_variables["width"]) / model(1500))
            print(f"New bkg scatter: {self.env_variables['bkg_scatter']}")

            max_photoreceptor_rf_size = max([self.env_variables['uv_photoreceptor_rf_size'],
                                             self.env_variables['red_photoreceptor_rf_size']])
            if "light_gradient" in self.env_variables:
                light_gradient = self.env_variables['light_gradient']
            else:
                light_gradient = 0

            max_visual_distance = (self.env_variables["width"] ** 2 + self.env_variables["height"] ** 2) ** 0.5
            if "max_visual_range" in self.env_variables:
                if self.env_variables["max_visual_range"]:
                    max_visual_distance = self.env_variables["max_visual_range"]

            self.board = NewDrawingBoard(self.env_variables['width'], self.env_variables['height'],
                                         decay_rate=self.env_variables['decay_rate'],
                                         photoreceptor_rf_size=max_photoreceptor_rf_size,
                                         using_gpu=using_gpu, visualise_mask=self.env_variables['visualise_mask'],
                                         prey_size=self.env_variables['prey_size'],
                                         predator_size=self.env_variables['predator_size'],
                                         visible_scatter=self.env_variables['bkg_scatter'],
                                         background_grating_frequency=self.env_variables[
                                             'background_grating_frequency'],
                                         dark_light_ratio=self.env_variables['dark_light_ratio'],
                                         dark_gain=self.env_variables['dark_gain'],
                                         light_gain=self.env_variables['light_gain'],
                                         red_occlusion_gain=self.env_variables['red_occlusion_gain'],
                                         uv_occlusion_gain=self.env_variables['uv_occlusion_gain'],
                                         red2_occlusion_gain=self.env_variables['red2_occlusion_gain'],
                                         light_gradient=light_gradient,
                                         max_visual_distance=max_visual_distance
                                         )
        else:
            self.board = DrawingBoard(self.env_variables['width'], self.env_variables['height'])
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
        self.predator_bodies = []
        self.predator_shapes = []

        self.predator_body = None
        self.predator_shape = None
        self.predator_target = None

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []

        self.last_action = None

        self.vegetation_bodies = []
        self.vegetation_shapes = []

        self.background = None

        self.prey_consumed_this_step = False

        self.predator_attacks_avoided = 0
        self.prey_caught = 0
        self.sand_grains_bumped = 0
        self.steps_near_vegetation = 0

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
        self.visualise_mask = self.env_variables['visualise_mask']
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

        # For visualisation of previous actions
        self.action_buffer = []
        self.position_buffer = []
        self.fish_angle_buffer = []

        # For logging
        self.failed_capture_attempts = 0
        self.in_light_history = []

    def reset(self):
        self.num_steps = 0
        self.fish.hungry = 0
        self.fish.stress = 1
        self.fish.touched_edge_this_step = False
        self.prey_caught = 0
        self.predator_attacks_avoided = 0
        self.sand_grains_bumped = 0
        self.steps_near_vegetation = 0
        self.energy_level_log = []
        if self.new_simulation:
            self.board.light_gain = self.env_variables["light_gain"]
            self.board.luminance_mask = self.board.get_luminance_mask(self.env_variables["dark_light_ratio"], self.env_variables["dark_gain"])

        # New energy system:
        self.fish.energy_level = 1

        for i, shp in enumerate(self.prey_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.sand_grain_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.vegetation_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.prey_cloud_wall_shapes):
            self.space.remove(shp)

        self.prey_cloud_wall_shapes = []

        if self.predator_shape is not None:
            self.remove_realistic_predator()

        if self.new_simulation:
            self.predator_location = None
            self.remaining_predator_attacks = None
            self.total_predator_steps = None
            self.new_attack_due = False
            # Reset salt gradient
            if self.env_variables["salt"]:
                self.reset_salt_gradient()
                self.fish.salt_health = 1.0
                self.salt_damage_history = []

            self.paramecia_gaits = []

            if self.env_variables["prey_reproduction_mode"]:
                self.prey_ages = []

            self.first_attack = False
            self.loom_predator_current_size = None

            self.board.reset()

        self.prey_shapes = []
        self.prey_bodies = []

        self.predator_shapes = []
        self.predator_bodies = []

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []

        self.vegetation_bodies = []
        self.vegetation_shapes = []

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
                if not self.check_proximity(cloud, self.env_variables["birth_rate_region_size"]/2):
                    new_location = (
                        np.random.randint(low=cloud[0] - (self.env_variables["birth_rate_region_size"] / 2),
                                          high=cloud[0] + (self.env_variables["birth_rate_region_size"] / 2)),
                        np.random.randint(low=cloud[1] - (self.env_variables["birth_rate_region_size"] / 2),
                                          high=cloud[1] + (self.env_variables["birth_rate_region_size"] / 2))
                    )
                    self.create_prey(new_location)
                    self.available_prey += 1

    def reset_salt_gradient(self):
        salt_source_x = np.random.randint(0, self.env_variables['width'] - 1)
        salt_source_y = np.random.randint(0, self.env_variables['height'] - 1)
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

    def output_frame(self, activations, internal_state, scale=0.25):
        # Adjust scale for larger environments

        # Saving mask frames (for debugging)
        if self.visualise_mask:
            frame = self.board.mask_buffer_time_point * 255.0
            frame = rescale(frame, scale, multichannel=True, anti_aliasing=True)
            self.mask_buffer.append(frame)
            self.board.mask_buffer_point = None

        if self.using_gpu:
            arena = copy.copy(self.board.db_visualisation.get() * 255.0)
        else:
            arena = copy.copy(self.board.db_visualisation * 255.0)

        arena[0, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[self.env_variables['height'] - 1, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[:, 0, 0] = np.ones(self.env_variables['height']) * 255
        arena[:, self.env_variables['width'] - 1, 0] = np.ones(self.env_variables['height']) * 255

        if self.new_simulation:
            empty_green_eyes = np.zeros((20, self.env_variables["width"], 1))
            eyes = self.fish.get_visual_inputs_new()
            eyes = np.concatenate((eyes[:, :, :1], empty_green_eyes, eyes[:, :, 1:2]),
                                  axis=2)  # Note removes second red channel.
        else:
            eyes = self.fish.get_visual_inputs()

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
                pos = (activations[ac] - adr[0]) / (adr[1] - adr[0])

                pos[pos < 0] = 0
                pos[pos > 1] = 1

                this_ac[:, :, 0] = resize(pos, (20, self.env_variables['width'])) * 255
                this_ac[:, :, 1] = resize(pos, (20, self.env_variables['width'])) * 255
                this_ac[:, :, 2] = resize(pos, (20, self.env_variables['width'])) * 255

                frame = np.vstack((frame, np.zeros((20, self.env_variables['width'], 3)), this_ac))

        if self.env_variables["show_action_space_usage"]:
            action_display = self.get_action_space_usage_display(frame.shape)
            frame = np.hstack((frame, np.zeros((frame.shape[0], 20, 3)), action_display))

        frame = rescale(frame, scale, multichannel=True, anti_aliasing=True)

        return frame

    def draw_shapes(self, visualisation):
        if visualisation:
            if self.env_variables["show_previous_actions"]:
                self._draw_past_actions(self.env_variables["show_previous_actions"])
            scaling_factor = 1500 / self.env_variables["width"]
            self._draw_shapes_environmental(visualisation, self.env_variables['prey_size_visualisation']/scaling_factor)
        else:
            self._draw_shapes_environmental(visualisation, self.env_variables['prey_size'])

    def _draw_past_actions(self, n_actions_to_show):
        # Select subset of actions to show
        if len(self.action_buffer) > n_actions_to_show:
            actions_to_show = self.action_buffer[len(self.action_buffer)-n_actions_to_show:]
            positions_to_show = self.position_buffer[len(self.position_buffer)-n_actions_to_show:]
            fish_angles_to_show = self.fish_angle_buffer[len(self.fish_angle_buffer)-n_actions_to_show:]
        else:
            actions_to_show = self.action_buffer
            positions_to_show = self.position_buffer
            fish_angles_to_show = self.fish_angle_buffer

        for i, a in enumerate(actions_to_show):
            adjusted_colour_index = ((1-self.env_variables["bkg_scatter"]) * (i+1)/len(actions_to_show)) + self.env_variables["bkg_scatter"]
            if self.continuous_actions:
                # action_colour = (1 * ((i+1)/len(actions_to_show)), 0, 0)
                if a[1] < 0:
                    action_colour = (adjusted_colour_index, self.env_variables["bkg_scatter"], self.env_variables["bkg_scatter"])
                else:
                    action_colour = (self.env_variables["bkg_scatter"], adjusted_colour_index, adjusted_colour_index)

                self.board.show_action_continuous(a[0], a[1], fish_angles_to_show[i], positions_to_show[i][0],
                                       positions_to_show[i][1], action_colour)
            else:
                action_colour = self.fish.get_action_colour(actions_to_show[i], adjusted_colour_index, self.env_variables["bkg_scatter"])
                self.board.show_action_discrete(fish_angles_to_show[i], positions_to_show[i][0],
                                       positions_to_show[i][1], action_colour)

    def _draw_shapes_environmental(self, visualisation, prey_size):
        if visualisation:  # Only draw fish if in visualisation mode
            if self.env_variables["show_fish_body_energy_state"]:
                fish_body_colour = (1 - self.fish.energy_level, self.fish.energy_level, 0)
            else:
                fish_body_colour = self.fish.head.color

            self.board.fish_shape(self.fish.body.position, self.env_variables['fish_mouth_size'],
                                  self.env_variables['fish_head_size'], self.env_variables['fish_tail_length'],
                                  self.fish.mouth.color, fish_body_colour, self.fish.body.angle)

        if len(self.prey_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.prey_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.prey_bodies])).astype(int)
            rrs, ccs = self.board.multi_circles(px, py, prey_size)
            rrs = np.clip(rrs, 0, self.env_variables["width"]-1)
            ccs = np.clip(ccs, 0, self.env_variables["height"]-1)

            try:
                if visualisation:
                    self.board.db_visualisation[rrs, ccs] = self.prey_shapes[0].color
                else:
                    self.board.db[rrs, ccs] = self.prey_shapes[0].color
            except IndexError:
                print(f"Index Error for: PX: {max(rrs.flatten())}, PY: {max(ccs.flatten())}")
                if max(rrs.flatten()) > self.env_variables['height']:
                    lost_index = np.argmax(py)
                elif max(ccs.flatten()) > self.env_variables['width']:
                    lost_index = np.argmax(px)
                else:
                    lost_index = 0
                    print(f"Fix needs to be tuned: PX: {max(px)}, PY: {max(py)}")
                self.prey_bodies.pop(lost_index)
                self.prey_shapes.pop(lost_index)
                self.draw_shapes(visualisation=visualisation)

        if len(self.sand_grain_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.sand_grain_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.sand_grain_bodies])).astype(int)
            if visualisation:
                rrs, ccs = self.board.multi_circles(px, py, prey_size)
            else:
                rrs, ccs = self.board.multi_circles(px, py, self.env_variables['sand_grain_size'])

            try:
                if visualisation:
                    self.board.db_visualisation[rrs, ccs] = self.sand_grain_shapes[0].color
                else:
                    self.board.db[rrs, ccs] = self.sand_grain_shapes[0].color
            except IndexError:
                print(f"Index Error for: RRS: {max(rrs.flatten())}, CCS: {max(ccs.flatten())}")
                if max(rrs.flatten()) > self.env_variables['width']:
                    lost_index = np.argmax(px)
                elif max(ccs.flatten()) > self.env_variables['height']:
                    lost_index = np.argmax(py)
                else:
                    lost_index = 0
                    print(f"Fix needs to be tuned: PX: {max(px)}, PY: {max(py)}")
                self.sand_grain_bodies.pop(lost_index)
                self.sand_grain_shapes.pop(lost_index)
                self.draw_shapes(visualisation=visualisation)

        for i, pr in enumerate(self.predator_bodies):
            self.board.circle(pr.position, self.env_variables['predator_size'], self.predator_shapes[i].color, visualisation)

        for i, pr in enumerate(self.vegetation_bodies):
            self.board.vegetation(pr.position, self.env_variables['vegetation_size'], self.vegetation_shapes[i].color, visualisation)

        if self.predator_body is not None:
            if self.first_attack:
                self.board.circle(self.predator_body.position, self.loom_predator_current_size,
                                  self.predator_shape.color, visualisation)
            else:
                self.board.circle(self.predator_body.position, self.env_variables['predator_size'],
                                  self.predator_shape.color, visualisation)

        # For displaying location of salt source
        if visualisation:
            if self.env_variables["salt"]:
                pass
                #self.board.show_salt_location(self.salt_location)

        # For creating a screen around prey to test.
        if self.background:
            if self.background == "Green":
                colour = (0, 1, 0)
            elif self.background == "Red":
                colour = (1, 0, 0)
            else:
                print("Invalid Background Colour")
                return
            self.board.create_screen(self.fish.body.position, self.env_variables["max_vis_dist"], colour)

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
        if self.new_simulation and not self.env_variables["wall_reflection"]:
            return self._touch_wall_new(arbiter, space, data)
        else:
            return self._touch_wall(arbiter, space, data)

    def _touch_wall(self, arbiter, space, data):
        # print(f"Fish touched wall: {self.fish.body.position}")
        new_position_x = self.fish.body.position[0]
        new_position_y = self.fish.body.position[1]

        if new_position_x < self.env_variables['wall_buffer_distance']:  # Wall d
            new_position_x = self.env_variables['wall_buffer_distance'] + self.env_variables["fish_head_size"] + \
                             self.env_variables["fish_tail_length"]
        elif new_position_x > self.env_variables['width'] - self.env_variables['wall_buffer_distance']:  # wall b
            new_position_x = self.env_variables['width'] - (
                    self.env_variables['wall_buffer_distance'] + self.env_variables["fish_head_size"] +
                    self.env_variables["fish_tail_length"])
        if new_position_y < self.env_variables['wall_buffer_distance']:  # wall a
            new_position_y = self.env_variables['wall_buffer_distance'] + self.env_variables["fish_head_size"] + \
                             self.env_variables["fish_tail_length"]
        elif new_position_y > self.env_variables['height'] - self.env_variables['wall_buffer_distance']:  # wall c
            new_position_y = self.env_variables['height'] - (
                    self.env_variables['wall_buffer_distance'] + self.env_variables["fish_head_size"] +
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

    def _touch_wall_new(self, arbiter, space, data):
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

    def create_prey(self, prey_position=None):
        self.prey_bodies.append(pymunk.Body(self.env_variables['prey_mass'], self.env_variables['prey_inertia']))
        self.prey_shapes.append(pymunk.Circle(self.prey_bodies[-1], self.env_variables['prey_size']))
        self.prey_shapes[-1].elasticity = 1.0
        self.prey_bodies[-1].angle = np.random.uniform(0, np.pi*2)
        if prey_position is None:
            if not self.env_variables["differential_prey"]:
                self.prey_bodies[-1].position = (
                    np.random.randint(
                        self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'] + self.env_variables[
                            'wall_buffer_distance'],
                        self.env_variables['width'] - (
                                self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'] +
                                self.env_variables['wall_buffer_distance'])),
                    np.random.randint(
                        self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'] + self.env_variables[
                            'wall_buffer_distance'],
                        self.env_variables['height'] - (
                                self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'] +
                                self.env_variables['wall_buffer_distance'])))
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
        all_prey_positions = np.array([pr.position for pr in self.prey_bodies])
        fish_position = self.fish.body.position
        within_x = (all_prey_positions[:, 0] > fish_position[0] - sensing_distance) * (all_prey_positions[:, 0] < fish_position[0] + sensing_distance)
        within_y = (all_prey_positions[:, 1] > fish_position[1] - sensing_distance) * (all_prey_positions[:, 1] < fish_position[1] + sensing_distance)
        within_range = within_x * within_y
        return within_range

    def move_prey(self, micro_step):
        if self.new_simulation:
            self._move_prey_new(micro_step)
        else:
            self._move_prey()

    def _move_prey(self):
        # Not, currently, a prey isn't guaranteed to try to escape if a loud predator is near, only if it was going to
        # move anyway. Should reconsider this in the future.
        to_move = np.where(np.random.rand(len(self.prey_bodies)) < self.env_variables['prey_impulse_rate'])[0]
        for ii in range(len(to_move)):
            if self.check_proximity(self.prey_bodies[to_move[ii]].position,
                                    self.env_variables['prey_sensing_distance']) and self.env_variables["prey_jump"]:
                self.prey_bodies[ii].angle = self.fish.body.angle + np.random.uniform(-1, 1)
                self.prey_bodies[to_move[ii]].apply_impulse_at_local_point((self.get_last_action_magnitude(), 0))
            else:
                adjustment = np.random.uniform(-self.env_variables['prey_max_turning_angle'],
                                               self.env_variables['prey_max_turning_angle'])
                self.prey_bodies[to_move[ii]].angle = self.prey_bodies[to_move[ii]].angle + adjustment
                self.prey_bodies[to_move[ii]].apply_impulse_at_local_point((self.env_variables['prey_impulse'], 0))

    def _move_prey_new(self, micro_step):
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
                # if self.env_variables["prey_jump"] and np.random.choice([0, 1], size=1,
                #                                                         p=[1 - self.env_variables["p_escape"]/self.env_variables["phys_steps_per_sim_step"],
                #                                                            self.env_variables["p_escape"]/self.env_variables["phys_steps_per_sim_step"]])[0] == 1:
                #     prey_body.apply_impulse_at_local_point((self.env_variables["jump_speed_paramecia"], 0))

            else:
                if micro_step == 0:
                    prey_body.angle = prey_body.angle + angle_changes[i]

                prey_body.apply_impulse_at_local_point((impulses[i], 0))

    def touch_prey(self, arbiter, space, data):
        if self.new_simulation:
            return self.touch_prey_new(arbiter, space, data)
        else:
            return self.touch_prey_old(arbiter, space, data)

    def touch_prey_new(self, arbiter, space, data):
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

    def touch_prey_old(self, arbiter, space, data):
        if self.fish.making_capture:
            for i, shp in enumerate(self.prey_shapes):
                if shp == arbiter.shapes[0]:
                    space.remove(shp, shp.body)
                    self.prey_shapes.remove(shp)
                    self.prey_bodies.remove(shp.body)
            self.prey_caught += 1
            self.fish.prey_consumed = True
            self.prey_consumed_this_step = True

            return False
        else:
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
        if self.new_simulation:
            self.predator_shapes[-1].color = (0, 1, 0)
            # Made green so still visible to us but not to fish.
        else:
            self.predator_shapes[-1].color = (0, 0, 1)
        self.predator_shapes[-1].collision_type = 5

        self.space.add(self.predator_bodies[-1], self.predator_shapes[-1])

    def move_predator(self):
        for pr in self.predator_bodies:
            dist_to_fish = np.sqrt(
                (pr.position[0] - self.fish.body.position[0]) ** 2 + (pr.position[1] - self.fish.body.position[1]) ** 2)

            if dist_to_fish < self.env_variables['predator_sensing_dist']:
                pr.angle = np.pi / 2 - np.arctan2(self.fish.body.position[0] - pr.position[0],
                                                  self.fish.body.position[1] - pr.position[1])
                pr.apply_impulse_at_local_point((self.env_variables['predator_chase_impulse'], 0))

            elif np.random.rand(1) < self.env_variables['predator_impulse_rate']:
                pr.angle = np.random.rand(1) * 2 * np.pi
                pr.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

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

    def create_realistic_predator(self):
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

        if self.new_simulation:
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
        else:
            self.predator_shape.color = (0, 0, 1)

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

    def _move_realistic_predator_old(self):
        if self.check_predator_at_target():
            self.remove_realistic_predator()
            self.predator_attacks_avoided += 1
            return
        if self.check_predator_outside_walls():
            self.remove_realistic_predator()
            return

        self.predator_body.angle = np.pi / 2 - np.arctan2(
            self.predator_target[0] - self.predator_body.position[0],
            self.predator_target[1] - self.predator_body.position[1])
        self.predator_body.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def _move_realistic_predator_new(self, micro_step):
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
            if self.predator_distance_to_fish() > self.env_variables["max_predator_reorient_distance"]:
                self.predator_target = np.array(self.fish.body.position)

            self.predator_body.angle = np.pi / 2 - np.arctan2(
                self.predator_target[0] - self.predator_body.position[0],
                self.predator_target[1] - self.predator_body.position[1])
            self.predator_body.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def move_realistic_predator(self, micro_step):
        if self.new_simulation:
            self._move_realistic_predator_new(micro_step)
        else:
            self._move_realistic_predator_old()

    def remove_realistic_predator(self, arbiter=None, space=None, data=None):
        if self.predator_body is not None:
            self.space.remove(self.predator_shape, self.predator_shape.body)
            self.predator_shape = None
            self.predator_body = None
            self.predator_location = None
            self.predator_target = None
            self.remaining_predator_attacks = None
            self.total_predator_steps = None
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
        self.sand_grain_bodies[-1].position = (
            np.random.randint(self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['width'] - (
                                      self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'])),
            np.random.randint(self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['height'] - (
                                      self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'])))

        if "sand_grain_colour" in self.env_variables:
            self.sand_grain_shapes[-1].color = self.env_variables["sand_grain_colour"]
        else:
            self.sand_grain_shapes[-1].color = (0, 0, 1)

        self.sand_grain_shapes[-1].collision_type = 4
        self.sand_grain_shapes[-1].filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.sand_grain_bodies[-1], self.sand_grain_shapes[-1])

    def touch_grain(self, arbiter, space, data):
        if self.last_action == 3:
            self.sand_grains_bumped += 1

    def get_last_action_magnitude(self):
        return self.fish.prev_action_impulse * self.env_variables[
            'displacement_scaling_factor']  # Scaled down both for mass effects and to make it possible for the prey to be caught.

    def displace_sand_grains(self):
        penalty = 0
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

                if "sand_grain_touch_penalty" in self.env_variables:
                    penalty -= self.env_variables["sand_grain_touch_penalty"]
        return penalty

    def create_vegetation(self):
        size = self.env_variables['vegetation_size']
        vertices = [(0, 0), (0, size), (size / 2, size - size / 3), (size, size), (size, 0), (size / 2, size / 3)]
        self.vegetation_bodies.append(pymunk.Body(body_type=pymunk.Body.STATIC))
        self.vegetation_shapes.append(pymunk.Poly(self.vegetation_bodies[-1], vertices))
        self.vegetation_bodies[-1].position = (
            np.random.randint(self.env_variables['vegetation_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['width'] - (
                                      self.env_variables['vegetation_size'] + self.env_variables['fish_mouth_size'])),
            np.random.randint(self.env_variables['vegetation_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['height'] - (
                                      self.env_variables['vegetation_size'] + self.env_variables['fish_mouth_size'])))
        self.vegetation_shapes[-1].color = (0, 1, 0)
        self.vegetation_shapes[-1].collision_type = 1
        self.vegetation_shapes[-1].friction = 1

        self.space.add(self.vegetation_bodies[-1], self.vegetation_shapes[-1])

    def check_fish_near_vegetation(self):
        vegetation_locations = [v.position for v in self.vegetation_bodies]
        fish_surrounding_area = [[self.fish.body.position[0] - self.env_variables['vegetation_effect_distance'],
                                  self.fish.body.position[0] + self.env_variables['vegetation_effect_distance']],
                                 [self.fish.body.position[1] - self.env_variables['vegetation_effect_distance'],
                                  self.fish.body.position[1] + self.env_variables['vegetation_effect_distance']]]
        for veg in vegetation_locations:
            is_in_area = fish_surrounding_area[0][0] <= veg[0] <= fish_surrounding_area[0][1] and \
                         fish_surrounding_area[1][0] <= veg[1] <= fish_surrounding_area[1][1]
            if is_in_area:
                self.steps_near_vegetation += 1
                return True
        return False
