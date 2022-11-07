import math

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pymunk

from Environment.base_environment import BaseEnvironment
from Environment.Fish.fish import Fish


class NaturalisticEnvironment(BaseEnvironment):

    def __init__(self, env_variables, realistic_bouts, new_simulation, using_gpu, draw_screen=False, fish_mass=None,
                 collisions=True, relocate_fish=None, num_actions=10):
        super().__init__(env_variables, draw_screen, new_simulation, using_gpu, num_actions)

        if using_gpu:
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

        # For calibrating observation scaling
        # self.mean_observation_vals = [[0, 0, 0]]
        # self.max_observation_vals = [[0, 0, 0]]

        # For currents (new simulation):
        if self.new_simulation:
            self.impulse_vector_field = None
            self.coordinates_in_current = None  # May be used to provide efficient checking. Although vector comp probably faster.
            self.create_current()
            self.capture_fraction = int(
                self.env_variables["phys_steps_per_sim_step"] * self.env_variables['fraction_capture_permitted'])
            self.capture_start = 1 #int((self.env_variables['phys_steps_per_sim_step'] - self.capture_fraction) / 2)
            self.capture_end = self.capture_start + self.capture_fraction

        self.paramecia_distances = []
        self.relocate_fish = relocate_fish
        self.impulse_against_fish_previous_step = None

        self.recent_cause_of_death = None

        # For producing a useful PCI
        self.available_prey = self.env_variables["prey_num"]
        self.vector_agreement = []

    def reset(self):
        # print(f"Mean R: {sum([i[0] for i in self.mean_observation_vals])/len(self.mean_observation_vals)}")
        # print(f"Mean UV: {sum([i[1] for i in self.mean_observation_vals])/len(self.mean_observation_vals)}")
        # print(f"Mean R2: {sum([i[2] for i in self.mean_observation_vals])/len(self.mean_observation_vals)}")
        #
        # print(f"Max R: {max([i[0] for i in self.max_observation_vals])}")
        # print(f"Max UV: {max([i[1] for i in self.max_observation_vals])}")
        # print(f"Max R2: {max([i[2] for i in self.max_observation_vals])}")
        # self.mean_observation_vals = [[0, 0, 0]]
        # self.max_observation_vals = [[0, 0, 0]]
        super().reset()
        self.fish.body.position = (np.random.randint(self.env_variables['fish_mouth_size'] + 40,
                                                     self.env_variables['width'] - (self.env_variables[
                                                                                        'fish_mouth_size'] + 40)),
                                   np.random.randint(self.env_variables['fish_mouth_size'] + 40,
                                                     self.env_variables['height'] - (self.env_variables[
                                                                                         'fish_mouth_size'] + 40)))
        self.fish.body.angle = np.random.random() * 2 * np.pi
        self.fish.body.velocity = (0, 0)
        if self.env_variables["current_setting"]:
            self.impulse_vector_field *= np.random.choice([-1, 1], size=1, p=[0.5, 0.5]).astype(float)
        self.fish.capture_possible = False

        if self.env_variables["differential_prey"]:
            self.prey_cloud_locations = [
                [np.random.randint(low=120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                   high=self.env_variables['width'] - (
                                           self.env_variables['prey_size'] + self.env_variables[
                                       'fish_mouth_size']) - 120),
                 np.random.randint(low=120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                   high=self.env_variables['height'] - (
                                           self.env_variables['prey_size'] + self.env_variables[
                                       'fish_mouth_size']) - 120)]
                for cloud in range(int(self.env_variables["prey_cloud_num"]))]

            if "fixed_prey_distribution" in self.env_variables:
                if self.env_variables["fixed_prey_distribution"]:
                    x_locations = np.linspace(
                        120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                        self.env_variables['width'] - (
                                self.env_variables['prey_size'] + self.env_variables['fish_mouth_size']) - 120,
                        math.ceil(self.env_variables["prey_cloud_num"] ** 0.5))
                    y_locations = np.linspace(
                        120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                        self.env_variables['width'] - (
                                self.env_variables['prey_size'] + self.env_variables['fish_mouth_size']) - 120,
                        math.ceil(self.env_variables["prey_cloud_num"] ** 0.5))

                    self.prey_cloud_locations = np.concatenate((np.expand_dims(x_locations, 1),
                                                                np.expand_dims(y_locations, 1)), axis=1)
                    self.prey_cloud_locations = self.prey_cloud_locations[:self.env_variables["prey_cloud_num"]]

            if not self.env_variables["prey_reproduction_mode"]:
                self.build_prey_cloud_walls()

        for i in range(int(self.env_variables['prey_num'])):
            self.create_prey()

        for i in range(self.env_variables['sand_grain_num']):
            self.create_sand_grain()

        for i in range(self.env_variables['vegetation_num']):
            self.create_vegetation()

        self.impulse_against_fish_previous_step = None
        self.recent_cause_of_death = None
        self.available_prey = self.env_variables["prey_num"]
        self.vector_agreement = []

    def show_new_channel_sectors(self, left_eye_pos, right_eye_pos):
        left_sectors, right_sectors = self.fish.get_all_sectors([left_eye_pos[0], left_eye_pos[1]],
                                                                [right_eye_pos[0], right_eye_pos[1]],
                                                                self.fish.body.angle)
        # l_top_left, l_bottom_left, l_top_right, l_bottom_right = self.fish.left_eye.get_corner_sectors(left_sectors)
        # r_top_left, r_bottom_left, r_top_right, r_bottom_right = self.fish.right_eye.get_corner_sectors(right_sectors)

        field = self.board.db
        plt.figure(figsize=(20, 20))
        plt.imshow(field)
        for i, sector in enumerate(right_sectors):
            # sector = self.fish.right_eye.get_extra_vertices(i, sector, r_top_left, r_bottom_left, r_top_right, r_bottom_right)
            # sector = sorted(sector, key=lambda x: x[0])
            patch = plt.Polygon(sector, closed=True, color="r", alpha=0.2)
            plt.gca().add_patch(patch)
        for i, sector in enumerate(left_sectors):
            # sector = self.fish.left_eye.get_extra_vertices(i, sector, l_top_left, l_bottom_left, l_top_right, l_bottom_right)
            # sector = sorted(sector, key=lambda x: x[0])
            patch = plt.Polygon(sector, color="b", alpha=0.2)
            plt.gca().add_patch(patch)
        plt.show()

    def bring_fish_in_bounds(self):
        # Resolve if fish falls out of bounds.
        if self.fish.body.position[0] < 4 or self.fish.body.position[1] < 4 or \
                self.fish.body.position[0] > self.env_variables["width"] - 4 or \
                self.fish.body.position[1] > self.env_variables["height"] - 4:
            new_position = pymunk.Vec2d(np.clip(self.fish.body.position[0], 6, self.env_variables["width"] - 7),
                                        np.clip(self.fish.body.position[1], 6, self.env_variables["height"] - 7))
            self.fish.body.position = new_position

    def simulation_step(self, action, save_frames, frame_buffer, activations, impulse):
        self.prey_consumed_this_step = False
        self.last_action = action

        # Visualisation
        if frame_buffer is None:
            frame_buffer = []
        if self.env_variables["show_previous_actions"]:
            self.action_buffer.append(action)
            self.fish_angle_buffer.append(self.fish.body.angle)
        self.position_buffer.append(np.array(self.fish.body.position))

        if impulse is not None:
            # To calculate calibration curve.
            reward = self.fish.try_impulse(impulse)
        else:
            reward = self.fish.take_action(action)

        # For impulse direction logging (current opposition metric)
        self.fish.impulse_vector_x = self.fish.prev_action_impulse * np.sin(self.fish.body.angle)
        self.fish.impulse_vector_y = self.fish.prev_action_impulse * np.cos(self.fish.body.angle)

        # Add policy helper reward to encourage proximity to prey.
        for ii in range(len(self.prey_bodies)):
            if self.check_proximity(self.prey_bodies[ii].position, self.env_variables['reward_distance']):
                reward += self.env_variables['proximity_reward']

        done = False

        # Change internal state variables
        self.fish.hungry += (1 - self.fish.hungry) * self.env_variables['hunger_inc_tau']
        self.fish.stress = self.fish.stress * self.env_variables['stress_compound']
        if self.predator_body is not None:
            self.fish.stress += 0.5

        self.init_predator()

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.move_prey(micro_step)
            reward += self.displace_sand_grains()
            if self.new_simulation:
                if self.env_variables["current_setting"]:
                    self.resolve_currents(micro_step)
                    self.bring_fish_in_bounds()
                if self.fish.making_capture and self.capture_start <= micro_step <= self.capture_end:
                    self.fish.capture_possible = True
                else:
                    self.fish.capture_possible = False
            if self.predator_body is not None:
                self.move_realistic_predator(micro_step)

            self.space.step(self.env_variables['phys_dt'])

            if self.fish.prey_consumed:
                if not self.env_variables["energy_state"]:
                    reward += self.env_variables['capture_basic_reward'] * self.fish.hungry
                self.fish.hungry *= self.env_variables['hunger_dec_tau']
                if len(self.prey_shapes) == 0:
                    done = True
                    self.recent_cause_of_death = "Prey-All-Eaten"

                self.fish.prey_consumed = False
            if self.fish.touched_edge:
                self.fish.touched_edge = False
            if self.fish.touched_predator:
                reward -= self.env_variables['predator_cost']
                done = True
                self.fish.touched_predator = False
                self.recent_cause_of_death = "Predator"

            if self.show_all:
                self.board.erase_visualisation(bkg=0.3)
                self.draw_shapes(visualisation=True)
                if self.draw_screen:
                    self.board_image.set_data(self.output_frame(activations, np.array([0, 0]), scale=0.5) / 255.)
                    plt.pause(0.0001)

        # Relocate fish (Assay mode only)
        if self.relocate_fish is not None:
            if self.relocate_fish[self.num_steps]:
                self.transport_fish(self.relocate_fish[self.num_steps])

        self.bring_fish_in_bounds()

        if self.new_simulation:
            # Energy level
            if self.env_variables["energy_state"]:
                reward = self.fish.update_energy_level(reward, self.prey_consumed_this_step)
                self.energy_level_log.append(self.fish.energy_level)
                if self.fish.energy_level < 0:
                    print("Fish ran out of energy")
                    done = True
                    self.recent_cause_of_death = "Starvation"

            # Salt health
            if self.env_variables["salt"]:
                salt_damage = self.salt_gradient[int(self.fish.body.position[0]), int(self.fish.body.position[1])]
                self.salt_damage_history.append(salt_damage)
                self.fish.salt_health = self.fish.salt_health + self.env_variables["salt_recovery"] - salt_damage
                if self.fish.salt_health > 1.0:
                    self.fish.salt_health = 1.0
                if self.fish.salt_health < 0:
                    print("Fish too salty")
                    done = True
                    self.recent_cause_of_death = "Salt"

                if "salt_reward_penalty" in self.env_variables:
                    if self.env_variables["salt_reward_penalty"] > 0 and salt_damage > self.env_variables[
                        "salt_recovery"]:
                        reward -= self.env_variables["salt_reward_penalty"] * salt_damage

            if self.predator_body is not None:
                self.total_predator_steps += 1

            if self.fish.touched_edge_this_step:
                reward -= self.env_variables["wall_touch_penalty"]
                self.fish.touched_edge_this_step = False

            if self.env_variables["prey_reproduction_mode"] and self.env_variables["differential_prey"]:
                self.reproduce_prey()
                self.prey_ages = [age + 1 for age in self.prey_ages]
                for i, age in enumerate(self.prey_ages):
                    if age > self.env_variables["prey_safe_duration"] and np.random.rand(1) < self.env_variables[
                        "p_prey_death"]:
                        # print("Removed prey")
                        self.remove_prey(i)
                        self.available_prey -= 1

        # Log whether or not fish in light
        self.in_light_history.append(self.fish.body.position[0] > self.dark_col)

        self.num_steps += 1
        self.board.erase(bkg=self.env_variables['bkg_scatter'])
        self.draw_shapes(visualisation=False)

        # Calculate internal state
        internal_state = []
        internal_state_order = []
        if self.env_variables['in_light']:
            internal_state.append(self.fish.body.position[0] > self.dark_col)
            internal_state_order.append("in_light")
        if self.env_variables['hunger']:
            internal_state.append(self.fish.hungry)
            internal_state_order.append("hunger")
        if self.env_variables['stress']:
            internal_state.append(self.fish.stress)
            internal_state_order.append("stress")
        if self.env_variables['energy_state']:
            internal_state.append(self.fish.energy_level)
            internal_state_order.append("energy_state")
        if self.env_variables['salt']:
            internal_state.append(salt_damage)
            internal_state_order.append("salt")
        if len(internal_state) == 0:
            internal_state.append(0)
        internal_state = np.array([internal_state])

        # OLD:
        # if self.env_variables['hunger'] and self.env_variables['stress']:
        #     internal_state = np.array([[in_light, self.fish.hungry, self.fish.stress]])
        # elif self.env_variables['hunger']:
        #     internal_state = np.array([[in_light, self.fish.hungry]])
        # elif self.env_variables['stress']:
        #     internal_state = np.array([[in_light, self.fish.stress]])
        # elif self.env_variables['energy_state']:
        #     internal_state = np.array([[in_light, self.fish.energy_level]])
        # else:
        #     internal_state = np.array([[in_light]])

        if self.new_simulation:
            observation, frame_buffer = self.resolve_visual_input_new(save_frames, activations, internal_state,
                                                                      frame_buffer)
        else:
            observation, frame_buffer = self.resolve_visual_input(save_frames, activations, internal_state,
                                                                  frame_buffer)

        # comb_obs = np.concatenate((observation[:, :, 0], observation[:, :, 1]), axis=0)
        # self.mean_observation_vals += [np.sum(comb_obs, axis=0)/len(comb_obs)]
        # self.max_observation_vals += [np.max(comb_obs, axis=0)]

        return observation, reward, internal_state, done, frame_buffer

    def init_predator(self):
        if self.new_simulation:
            if self.predator_location is None and np.random.rand(1) < self.env_variables["probability_of_predator"] and \
                    self.num_steps > self.env_variables['immunity_steps'] and not self.check_fish_near_vegetation() \
                    and not self.check_fish_not_near_wall():
                self.create_realistic_predator()
        else:
            if self.predator_shape is None and np.random.rand(1) < self.env_variables["probability_of_predator"] and \
                    self.num_steps > self.env_variables['immunity_steps'] and not self.check_fish_near_vegetation() \
                    and not self.check_fish_not_near_wall():
                self.create_realistic_predator()

    def resolve_visual_input_new(self, save_frames, activations, internal_state, frame_buffer):
        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        if self.predator_body is not None:
            predator_bodies = np.array([self.predator_body.position])
        else:
            predator_bodies = np.array([])

        full_masked_image = self.board.get_masked_pixels(np.array(self.fish.body.position),
                                                         np.array([i.position for i in self.prey_bodies]),
                                                         predator_bodies
                                                         )

        self.fish.left_eye.read(full_masked_image, left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(full_masked_image, right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        if save_frames or self.draw_screen:
            self.board.erase_visualisation(bkg=0.3)
            self.draw_shapes(visualisation=True)
            relative_dark_gain = self.env_variables["dark_gain"] / self.env_variables["light_gain"]
            self.board.apply_light(self.dark_col, relative_dark_gain, 1, visualisation=True)

            if self.env_variables['show_channel_sectors']:
                self.fish.left_eye.show_points(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
                self.fish.right_eye.show_points(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

            if save_frames:
                scaling_factor = 1500 / self.env_variables["width"]
                frame = self.output_frame(activations, internal_state, scale=0.25 * scaling_factor)
                frame_buffer.append(frame)
            if self.draw_screen:
                frame = self.output_frame(activations, internal_state, scale=0.5) / 255.
                self.board_image.set_data(frame/np.max(frame))
                plt.pause(0.000001)

        # observation = self.chosen_math_library.dstack((self.fish.left_eye.readings,
        #                                                self.fish.right_eye.readings))
        observation = np.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))
        # self.plot_observation(observation)
        # distance = ((self.fish.body.position[0]-self.prey_bodies[-1].position[0])**2 +
        #             (self.fish.body.position[1]-self.prey_bodies[-1].position[1])**2) ** 0.5
        # print(f"Prey Distance: {distance}\n")
        # self.paramecia_distances.append(distance)

        # if self.using_gpu:
        #     return observation.get(), frame_buffer
        # else:
        #     return observation, frame_buffer

        return observation, frame_buffer

    def plot_observation(self, observation):
        if self.using_gpu:
            observation = observation.get()
        observation = np.concatenate(
            (observation[:, 0:1, :], np.zeros((observation.shape[0], 1, observation.shape[2])), observation[:, 1:2, :]),
            axis=1)

        left_1 = observation[:, :, 0]
        right_1 = observation[:, :, 1]

        left_1 = np.expand_dims(left_1, 0)
        right_1 = np.expand_dims(right_1, 0)
        fig, axs = plt.subplots(2, 1, sharex=True)

        axs[0].imshow(left_1, aspect="auto")
        axs[0].set_ylabel("Left eye")
        axs[1].imshow(right_1, aspect="auto")
        axs[1].set_ylabel("Right eye")
        axs[1].set_xlabel("Photoreceptor")
        plt.show()

    def resolve_visual_input(self, save_frames, activations, internal_state, frame_buffer):
        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        self.fish.left_eye.read(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        if save_frames or self.draw_screen:
            self.board.erase_visualisation(bkg=0.3)
            self.draw_shapes(visualisation=True)
            relative_dark_gain = self.env_variables["dark_gain"] / self.env_variables["light_gain"]
            self.board.apply_light(self.dark_col, relative_dark_gain, 1, visualisation=True)
            self.fish.left_eye.show_points(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
            self.fish.right_eye.show_points(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)
            plt.imshow(self.board.db)
            plt.show()
            if save_frames:
                frame_buffer.append(self.output_frame(activations, internal_state, scale=0.25))
            if self.draw_screen:
                self.board_image.set_data(self.output_frame(activations, internal_state, scale=0.5) / 255.)
                plt.pause(0.000001)

        observation = np.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))
        return observation

    def create_current(self):
        """
        Creates relevant matrices given defined currents.

        Modes:
        - Circular - Parameters:
           * unit diameter (relative to environment, must be <1).
           * max current strength
           * Current variance/decay
           * Max_current_width
        - Diagonal (could be with or without dispersal)
        """
        if self.env_variables["current_setting"] == "Circular":
            print("Creating circular current")
            self.create_circular_current()
        elif self.env_variables["current_setting"] == "Linear":
            print("Creating linear current")
            self.create_linear_current()
        # elif self.env_variables["current_setting"] == "Diagonal":
        #     self.create_diagonal_current()
        else:
            print("No current specified.")
            # self.impulse_vector_field = np.zeros((self.env_variables["width"], self.env_variables["height"], 2))

    def create_circular_current(self):
        unit_circle_radius = self.env_variables["unit_circle_diameter"] / 2
        circle_width = self.env_variables['current_width']
        circle_variance = self.env_variables['current_strength_variance']
        max_current_strength = self.env_variables['max_current_strength']

        arena_center = np.array([self.env_variables["width"] / 2, self.env_variables["height"] / 2])

        # All coordinates:
        xp, yp = np.arange(self.env_variables["width"]), np.arange(self.env_variables["height"])
        xy, yp = np.meshgrid(xp, yp)
        xy = np.expand_dims(xy, 2)
        yp = np.expand_dims(yp, 2)
        all_coordinates = np.concatenate((xy, yp), axis=2)
        relative_coordinates = all_coordinates - arena_center  # TO compute coordinates relative to position in center
        distances_from_center = (relative_coordinates[:, :, 0] ** 2 + relative_coordinates[:, :, 1] ** 2) ** 0.5
        distances_from_center = np.expand_dims(distances_from_center, 2)

        xy1 = relative_coordinates[:, :, 0]
        yp1 = relative_coordinates[:, :, 1]
        u = -yp1 / np.sqrt(xy1 ** 2 + yp1 ** 2)
        v = xy1 / np.sqrt(xy1 ** 2 + yp1 ** 2)
        # u, v = np.meshgrid(u, v)
        u = np.expand_dims(u, 2)
        v = np.expand_dims(v, 2)
        vector_field = np.concatenate((u, v), axis=2)

        ### Impose ND structure
        # Compute distance from center at each point
        absolute_distances_from_center = np.absolute(distances_from_center[:, :, 0])
        normalised_distance_from_center = absolute_distances_from_center / np.max(absolute_distances_from_center)
        distance_from_talweg = normalised_distance_from_center - unit_circle_radius
        distance_from_talweg = np.abs(distance_from_talweg)
        distance_from_talweg = np.expand_dims(distance_from_talweg, 2)
        talweg_closeness = 1 - distance_from_talweg
        talweg_closeness = (talweg_closeness ** 2) * circle_variance
        current_strength = (talweg_closeness / np.max(talweg_closeness)) * max_current_strength
        current_strength = current_strength[:, :, 0]

        # (Distances - optimal_distance). This forms a subtraction matrix which can be related to the variance.
        adjusted_normalised_distance_from_center = normalised_distance_from_center ** 2

        ### Set cutoffs to 0 outside width
        inside_radius2 = (unit_circle_radius - (circle_width / 2)) ** 2
        outside_radius2 = (unit_circle_radius + (circle_width / 2)) ** 2
        inside = inside_radius2 < adjusted_normalised_distance_from_center
        outside = adjusted_normalised_distance_from_center < outside_radius2
        within_current = inside * outside * 1
        current_strength = current_strength * within_current

        # Scale vector field
        current_strength = np.expand_dims(current_strength, 2)
        vector_field = current_strength * vector_field

        # Prevent middle index being Nan, which causes error.
        vector_field[int(self.env_variables["width"] / 2), int(self.env_variables["height"] / 2)] = 0

        self.impulse_vector_field = vector_field

        # plt.streamplot(xy[:, :, 0], yp[:, :, 0], vector_field[:, :, 0], vector_field[:, :, 1])
        # plt.show()

    def create_diagonal_current(self):
        ...

    def create_linear_current(self):
        current_width = self.env_variables['current_width'] * self.env_variables["height"]
        current_variance = self.env_variables['current_strength_variance']
        max_current_strength = self.env_variables['max_current_strength']

        # Create vector field of same vectors moving in x direction
        vector = np.array([1, 0])
        vector_field = np.tile(vector, (self.env_variables["width"], self.env_variables["height"], 1))

        # Scale as distance from y=h/2
        centre = self.env_variables["height"] / 2
        xp, yp = np.arange(self.env_variables["width"]), np.arange(self.env_variables["height"])
        xy, yp = np.meshgrid(xp, yp)
        # all_coordinates = np.concatenate((xy, yp), axis=2)
        relative_y_coordinates = yp - centre
        relative_y_coordinates = np.absolute(relative_y_coordinates)
        closeness_to_centre = centre - relative_y_coordinates
        closeness_to_centre = (closeness_to_centre ** 2) * current_variance
        current_strength = (closeness_to_centre / np.max(closeness_to_centre)) * max_current_strength

        # Cut off outside of width
        upper_cut_off = int(centre - (current_width / 2))
        lower_cut_off = int(centre + (current_width / 2))
        current_strength[lower_cut_off:, :] = 0
        current_strength[:upper_cut_off, :] = 0
        current_strength = np.expand_dims(current_strength, 2)

        # Scale vector field
        vector_field = vector_field * current_strength
        self.impulse_vector_field = vector_field

        # plt.streamplot(xy[:, :], yp[:, :], vector_field[:, :, 0], vector_field[:, :, 1])
        # plt.ylim(0, 3000)
        # plt.xlim(0, 3000)
        # plt.show()


    def resolve_currents(self, micro_step):
        """Currents act on prey and fish."""
        all_feature_coordinates = np.array(
            [self.fish.body.position] + [body.position for body in self.prey_bodies]).astype(int)
        original_orientations = np.array([self.fish.body.angle] + [body.angle for body in self.prey_bodies])

        try:
            associated_impulse_vectors = self.impulse_vector_field[
                all_feature_coordinates[:, 0], all_feature_coordinates[:, 1]]
        except:
            print("Feature coordinates out of range, exception thrown.")
            print(f"Feature coordinates: {all_feature_coordinates}")
            associated_impulse_vectors = np.array([[0.0, 0.0] for i in range(all_feature_coordinates.shape[0])])

        self.fish.body.angle = np.pi
        self.fish.body.apply_impulse_at_local_point(
            (associated_impulse_vectors[0, 1], associated_impulse_vectors[0, 0]))

        for i, vector in enumerate(associated_impulse_vectors[1:]):
            self.prey_bodies[i].angle = np.pi
            self.prey_bodies[i].apply_impulse_at_local_point((vector[1], vector[0]))
            self.prey_bodies[i].angle = original_orientations[i + 1]
        self.fish.body.angle = original_orientations[0]

        # Add to log about swimming against currents...
        self.impulse_against_fish_previous_step = [associated_impulse_vectors[0, 1], associated_impulse_vectors[0, 0]]

        if micro_step == 0:
            # Log fish-current vector agreement
            self.vector_agreement.append((self.fish.impulse_vector_x * associated_impulse_vectors[0, 1]) + \
                                         (self.fish.impulse_vector_y * associated_impulse_vectors[0, 0]) * 5)


    def transport_fish(self, target_feature):
        """In assay mode only, relocates fish to a target feature from the following options:
           C: Near Prey cluster
           E: Away from any prey cluster and walls
           W: Near walls, with them in view of both eyes.
           P: Adds a predator nearby (one step away from capture)
        """
        if target_feature == "C":
            chosen_cluster = np.random.choice(range(len(self.prey_cloud_locations)))
            cluster_coordinates = self.prey_cloud_locations[chosen_cluster]
            self.fish.body.position = np.array(cluster_coordinates)
        elif target_feature == "E":
            xp, yp = np.arange(200, self.env_variables["width"] - 200), np.arange(200,
                                                                                  self.env_variables["height"] - 200)
            xy, yp = np.meshgrid(xp, yp)
            xy = np.expand_dims(xy, 2)
            yp = np.expand_dims(yp, 2)
            all_coordinates = np.concatenate((xy, yp), axis=2)
            all_prey_locations = [b.position for b in self.prey_bodies]
            for p in all_prey_locations:
                all_coordinates[int(p[0] - 100): int(p[0] + 100), int(p[1] - 100): int(p[1] + 100)] = False
            suitable_locations = all_coordinates.reshape(-1, all_coordinates.shape[-1])
            suitable_locations = [c for c in suitable_locations if c[0] != 0]
            choice = np.random.choice(range(len(suitable_locations)))
            location_away = suitable_locations[choice]
            self.fish.body.position = np.array(location_away)
