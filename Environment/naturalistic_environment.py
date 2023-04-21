import math

import numpy as np
import matplotlib.pyplot as plt
import pymunk

from Environment.base_environment import BaseEnvironment
from Environment.Fish.fish import Fish
from Analysis.Behavioural.Tools.get_fish_prey_incidence import get_fish_prey_incidence_multiple_prey
from matplotlib.animation import FFMpegWriter


class NaturalisticEnvironment(BaseEnvironment):

    def __init__(self, env_variables, using_gpu, relocate_fish=None, num_actions=10,
                 run_version=None, split_event=None, modification=None):
        super().__init__(env_variables, using_gpu, num_actions)

        if using_gpu:
            import cupy as cp
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

        # For currents (new simulation):
        self.impulse_vector_field = None
        self.coordinates_in_current = None  # May be used to provide efficient checking. Although vector comp probably faster.
        self.create_current()
        self.capture_fraction = int(
            self.env_variables["phys_steps_per_sim_step"] * self.env_variables['fraction_capture_permitted'])
        self.capture_start = 1  # int((self.env_variables['phys_steps_per_sim_step'] - self.capture_fraction) / 2)
        self.capture_end = self.capture_start + self.capture_fraction

        self.paramecia_distances = []
        self.relocate_fish = relocate_fish
        self.impulse_against_fish_previous_step = None

        self.recent_cause_of_death = None

        # For producing a useful PCI
        self.vector_agreement = []

        # For producing useful PAI
        self.total_predators_survived = 0

        # For Reward tracking (debugging)
        self.energy_associated_reward = 0
        self.action_associated_reward = 0
        self.salt_associated_reward = 0
        self.predator_associated_reward = 0
        self.wall_associated_reward = 0
        self.sand_grain_associated_reward = 0

        self.assay_run_version = run_version
        self.split_event = split_event
        self.modification = modification

    def reset(self):
        # print (f"Mean R: {sum([i[0] for i in self.mean_observation_vals])/len(self.mean_observation_vals)}")
        # print(f"Mean UV: {sum([i[1] for i in self.mean_observation_vals])/len(self.mean_observation_vals)}")
        # print(f"Mean R2: {sum([i[2] for i in self.mean_observation_vals])/len(self.mean_observation_vals)}")
        #
        # print(f"Max R: {max([i[0] for i in self.max_observation_vals])}")
        # print(f"Max UV: {max([i[1] for i in self.max_observation_vals])}")
        # print(f"Max R2: {max([i[2] for i in self.max_observation_vals])}")
        # self.mean_observation_vals = [[0, 0, 0]]
        # self.max_observation_vals = [[0, 0, 0]]
        super().reset()
        self.fish.body.position = (np.random.randint(self.env_variables['fish_mouth_radius'] + 40,
                                                     self.env_variables['arena_width'] - (self.env_variables[
                                                                                              'fish_mouth_radius'] + 40)),
                                   np.random.randint(self.env_variables['fish_mouth_radius'] + 40,
                                                     self.env_variables['arena_height'] - (self.env_variables[
                                                                                               'fish_mouth_radius'] + 40)))
        self.fish.body.angle = np.random.random() * 2 * np.pi

        self.fish.body.velocity = (0, 0)
        if self.env_variables["current_setting"]:
            self.impulse_vector_field *= np.random.choice([-1, 1], size=1, p=[0.5, 0.5]).astype(float)
        self.fish.capture_possible = False

        if self.env_variables["differential_prey"]:
            self.prey_cloud_locations = [
                [np.random.randint(
                    low=120 + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                    high=self.env_variables['arena_width'] - (
                            self.env_variables['prey_radius'] + self.env_variables[
                        'fish_mouth_radius']) - 120),
                 np.random.randint(
                     low=120 + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                     high=self.env_variables['arena_height'] - (
                             self.env_variables['prey_radius'] + self.env_variables[
                         'fish_mouth_radius']) - 120)]
                for cloud in range(int(self.env_variables["prey_cloud_num"]))]

            self.sand_grain_cloud_locations = [
                [np.random.randint(
                    low=120 + self.env_variables['sand_grain_radius'] + self.env_variables['fish_mouth_radius'],
                    high=self.env_variables['arena_width'] - (
                            self.env_variables['sand_grain_radius'] + self.env_variables[
                        'fish_mouth_radius']) - 120),
                    np.random.randint(
                        low=120 + self.env_variables['sand_grain_radius'] + self.env_variables['fish_mouth_radius'],
                        high=self.env_variables['arena_height'] - (
                                self.env_variables['sand_grain_radius'] + self.env_variables[
                            'fish_mouth_radius']) - 120)]
                for cloud in range(int(self.env_variables["sand_grain_num"]))]

            if "fixed_prey_distribution" in self.env_variables:
                if self.env_variables["fixed_prey_distribution"]:
                    x_locations = np.linspace(
                        120 + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                        self.env_variables['arena_width'] - (
                                self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius']) - 120,
                        math.ceil(self.env_variables["prey_cloud_num"] ** 0.5))
                    y_locations = np.linspace(
                        120 + self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius'],
                        self.env_variables['arena_width'] - (
                                self.env_variables['prey_radius'] + self.env_variables['fish_mouth_radius']) - 120,
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

        self.impulse_against_fish_previous_step = None
        self.recent_cause_of_death = None
        self.available_prey = self.env_variables["prey_num"]
        self.vector_agreement = []

        self.total_predators = 0
        self.total_predators_survived = 0

        # For Reward tracking (debugging)
        print(f"""REWARD CONTRIBUTIONS:        
Energy: {self.energy_associated_reward}
Action: {self.action_associated_reward}
Salt: {self.salt_associated_reward}
Predator: {self.predator_associated_reward}
Wall: {self.wall_associated_reward}
Sand grain: {self.sand_grain_associated_reward}
""")

        self.energy_associated_reward = 0
        self.action_associated_reward = 0
        self.salt_associated_reward = 0
        self.predator_associated_reward = 0
        self.wall_associated_reward = 0
        self.sand_grain_associated_reward = 0

    def load_simulation(self, buffer, sediment, energy_state):
        self.num_steps = len(buffer.fish_position_buffer)

        self.board.global_sediment_grating = self.chosen_math_library.array(np.expand_dims(sediment, 2))

        self.salt_location = buffer.salt_location
        self.reset_salt_gradient(buffer.salt_location)
        self.clear_environmental_features()

        # Create prey in proper positions and orientations
        final_step_prey_positions = buffer.prey_positions_buffer[-1]
        final_step_prey_orientations = buffer.prey_orientations_buffer[-1]
        for p, o in zip(final_step_prey_positions, final_step_prey_orientations):
            if p[0] != 10000.0:
                self.create_prey(prey_position=p, prey_orientation=o)

        self.prey_ages = np.array(buffer.prey_age_buffer[-1]).astype(int)
        self.paramecia_gaits = np.array(buffer.prey_gait_buffer[-1]).astype(int)

        # Create predators in proper position and orientation.
        final_step_predator_position = buffer.predator_position_buffer[-1]
        final_step_predator_orientation = buffer.predator_orientation_buffer[-1]
        if final_step_predator_position[0] != 10000.0:

            # Find step when predator was introduced. Get fish position then.
            predator_present = (np.array(buffer.predator_position_buffer)[:, 0] != 10000.0)
            predator_lifespan = 0
            for p in reversed(predator_present):
                if p:
                    predator_lifespan += 1
                else:
                    break
            predator_target = buffer.fish_position_buffer[-predator_lifespan]

            self.load_predator(predator_position=final_step_predator_position,
                               predator_orientation=final_step_predator_orientation,
                               predator_target=predator_target)

        self.fish.body.position = np.array(buffer.fish_position_buffer[-1])
        self.fish.body.angle = np.array(buffer.fish_angle_buffer[-1])
        self.fish.energy_level = energy_state

        # Get latest observation.
        self.board.FOV.update_field_of_view(self.fish.body.position)
        self.draw_walls_and_sediment()
        observation, full_masked_image = self.resolve_visual_input()
        return observation

    def check_condition_met(self):
        """For the split assay mode - checks whether the specified condition is met at each step"""

        if self.split_event == "One-Prey-Close":

            if self.num_steps == 100:
                return True

            if len(self.prey_bodies) > 0:
                max_angular_deviation = np.pi / 2  # Anywhere in visual field.
                max_distance = 100  # 10mm

                prey_near = self.check_proximity_all_prey(sensing_distance=max_distance)
                fish_prey_incidence = self.get_fish_prey_incidence()
                within_visual_field = np.absolute(fish_prey_incidence) < max_angular_deviation

                prey_close = prey_near * within_visual_field
                num_prey_close = np.sum(prey_close * 1)
                if num_prey_close == 1:
                    return True
        elif self.split_event == "Empty-Surroundings":
            ...
        else:
            print(self.split_event)
            print("Invalid Split Event Entered")

        return False

    def make_modification(self):
        # Note of conditions to impose: Remove nearby prey, add nearby prey.
        if self.modification == "Nearby-Prey-Removal":
            max_angular_deviation = np.pi / 2
            max_distance = 100  # 10mm

            prey_near = self.check_proximity_all_prey(sensing_distance=max_distance)
            fish_prey_incidence = self.get_fish_prey_incidence()
            within_visual_field = np.absolute(fish_prey_incidence) < max_angular_deviation

            prey_close = prey_near * within_visual_field

            prey_index_offset = len(self.prey_bodies)
            for i, p in enumerate(reversed(prey_close[0][0])):
                if p:
                    self.remove_prey(prey_index_offset - i)
                    print("Removed prey due to modification")

        else:
            print("Invalid Modification Entered")

    def bring_fish_in_bounds(self):
        # Resolve if fish falls out of bounds.
        if self.fish.body.position[0] < 4 or self.fish.body.position[1] < 4 or \
                self.fish.body.position[0] > self.env_variables["arena_width"] - 4 or \
                self.fish.body.position[1] > self.env_variables["arena_height"] - 4:
            new_position = pymunk.Vec2d(np.clip(self.fish.body.position[0], 6, self.env_variables["arena_width"] - 30),
                                        np.clip(self.fish.body.position[1], 6, self.env_variables["arena_height"] - 30))
            self.fish.body.position = new_position

    def simulation_step(self, action, impulse):
        self.prey_consumed_this_step = False
        self.last_action = action
        self.fish.touched_sand_grain = False

        # Visualisation
        self.action_buffer.append(action)
        self.fish_angle_buffer.append(self.fish.body.angle)
        self.position_buffer.append(np.array(self.fish.body.position))

        reward = self.fish.take_action(action)

        # For Reward tracking (debugging)
        self.action_associated_reward += reward

        # For impulse direction logging (current opposition metric)
        self.fish.impulse_vector_x = self.fish.prev_action_impulse * np.sin(self.fish.body.angle)
        self.fish.impulse_vector_y = self.fish.prev_action_impulse * np.cos(self.fish.body.angle)

        # Add policy helper reward to encourage proximity to prey.
        # for ii in range(len(self.prey_bodies)):
        #     if self.check_proximity(self.prey_bodies[ii].position, self.env_variables['reward_distance']):
        #         reward += self.env_variables['proximity_reward']

        done = False

        # Change internal state variables
        self.fish.stress = self.fish.stress * self.env_variables['stress_compound']
        if self.predator_body is not None:
            self.fish.stress += 0.5

        self.init_predator()

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.move_prey(micro_step)
            self.displace_sand_grains()

            if self.env_variables["current_setting"]:
                self.bring_fish_in_bounds()
                self.resolve_currents(micro_step)
            if self.fish.making_capture and self.capture_start <= micro_step <= self.capture_end:
                self.fish.capture_possible = True
            else:
                self.fish.capture_possible = False

            if self.predator_body is not None:
                self.move_predator()

            self.space.step(self.env_variables['phys_dt'])

            if self.fish.prey_consumed:
                if len(self.prey_shapes) == 0:
                    done = True
                    self.recent_cause_of_death = "Prey-All-Eaten"

                self.fish.prey_consumed = False
            if self.fish.touched_edge:
                self.fish.touched_edge = False

        if self.fish.touched_predator:
            print("Fish eaten by predator")
            reward -= self.env_variables['predator_cost']
            self.survived_attack = False
            self.predator_associated_reward -= self.env_variables["predator_cost"]
            self.remove_predator()
            self.fish.touched_predator = False

            # self.recent_cause_of_death = "Predator"
            # done = True

        if (self.predator_body is None) and self.survived_attack:
            print("Survived attack...")
            reward += self.env_variables["predator_avoidance_reward"]
            self.predator_associated_reward += self.env_variables["predator_cost"]
            self.survived_attack = False
            self.total_predators_survived += 1

        if self.fish.touched_sand_grain:
            reward -= self.env_variables["sand_grain_touch_penalty"]
            self.sand_grain_associated_reward -= self.env_variables["sand_grain_touch_penalty"]

        # Relocate fish (Assay mode only)
        if self.relocate_fish is not None:
            if self.relocate_fish[self.num_steps]:
                self.transport_fish(self.relocate_fish[self.num_steps])

        self.bring_fish_in_bounds()

        # Energy level
        if self.env_variables["energy_state"]:
            old_reward = reward
            reward = self.fish.update_energy_level(reward, self.prey_consumed_this_step)
            self.energy_associated_reward += reward - old_reward

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
                pass
                # done = True
                # self.recent_cause_of_death = "Salt"

            if self.env_variables["salt_reward_penalty"] > 0:  # and salt_damage > self.env_variables["salt_recovery"]:  TODO: Trying without this for simplicity
                reward -= self.env_variables["salt_reward_penalty"] * salt_damage
                self.salt_associated_reward -= self.env_variables['salt_reward_penalty'] * salt_damage
        else:
            salt_damage = 0

        if self.predator_body is not None:
            self.total_predator_steps += 1

        if self.fish.touched_edge_this_step:
            reward -= self.env_variables["wall_touch_penalty"]
            self.wall_associated_reward -= self.env_variables["wall_touch_penalty"]

            self.fish.touched_edge_this_step = False

        if self.env_variables["prey_reproduction_mode"] and self.env_variables["differential_prey"]:
            self.reproduce_prey()
            self.prey_ages = [age + 1 for age in self.prey_ages]
            for i, age in enumerate(self.prey_ages):
                if age > self.env_variables["prey_safe_duration"] and\
                        np.random.rand(1) < self.env_variables["p_prey_death"]:
                    if not self.check_proximity(self.prey_bodies[i].position, 200):
                        # print("Removed prey")
                        self.remove_prey(i)
                        self.available_prey -= 1

        # Log whether fish in light
        self.in_light_history.append(self.fish.body.position[0] > self.dark_col)

        self.num_steps += 1

        # Drawing the features visible at this step:
        self.board.FOV.update_field_of_view(self.fish.body.position)
        self.draw_walls_and_sediment()

        # Calculate internal state
        internal_state = []
        internal_state_order = []
        if self.env_variables['in_light']:
            internal_state.append(self.fish.body.position[0] > self.dark_col)
            internal_state_order.append("in_light")
        if self.env_variables['stress']:
            internal_state.append(self.fish.stress)
            internal_state_order.append("stress")
        if self.env_variables['energy_state']:
            internal_state.append(self.fish.energy_level)
            internal_state_order.append("energy_state")
        if self.env_variables['salt']:
            # Scale salt damage so is within same range as pixel counts going in (learning using these also failed with
            # lower scaling)
            # internal_state.append(0.0)
            if self.env_variables["max_salt_damage"] > 0:
                internal_state.append((255 * salt_damage)/self.env_variables["max_salt_damage"])
            else:
                internal_state.append(0.0)

            internal_state_order.append("salt")
        if len(internal_state) == 0:
            internal_state.append(0)
        internal_state = np.array([internal_state])


        if self.assay_run_version == "Original":
            if self.check_condition_met():
                print(f"Split condition met at step: {self.num_steps}")
                done = True
                self.switch_step = self.num_steps

        observation, full_masked_image = self.resolve_visual_input()

        return observation, reward, internal_state, done, full_masked_image

    def init_predator(self):
        if self.predator_location is None and \
                np.random.rand(1) < self.env_variables["probability_of_predator"] and \
                self.num_steps > self.env_variables['immunity_steps'] and \
                not self.check_fish_not_near_wall():

            self.create_predator()

    def resolve_visual_input(self):
        # eye positions within FOV - Relative eye positions to FOV
        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'] + self.board.max_visual_distance,
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'] + self.board.max_visual_distance)
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'] + self.board.max_visual_distance,
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables[
                'eyes_biasx'] + self.board.max_visual_distance)

        if self.predator_body is not None:
            predator_bodies = np.array([self.predator_body.position])
        else:
            predator_bodies = np.array([])

        prey_locations = [i.position for i in self.prey_bodies]
        sand_grain_locations = [i.position for i in self.sand_grain_bodies]
        full_masked_image, lum_mask = self.board.get_masked_pixels(np.array(self.fish.body.position),
                                                                   np.array(prey_locations + sand_grain_locations),
                                                                   predator_bodies)

        # Convert to FOV coordinates (to match eye coordinates)
        if len(prey_locations) > 0:
            prey_locations_array = np.array(prey_locations) - np.array(self.fish.body.position) + self.board.max_visual_distance
        else:
            prey_locations_array = np.array([])
        if len(sand_grain_locations) > 0:
            sand_grain_locations_array = np.array(sand_grain_locations) - np.array(
                self.fish.body.position) + self.board.max_visual_distance
        else:
            sand_grain_locations_array = np.empty((0, 2))

        self.fish.left_eye.read(full_masked_image, left_eye_pos[0], left_eye_pos[1], self.fish.body.angle, lum_mask,
                                prey_locations_array, sand_grain_locations_array)
        self.fish.right_eye.read(full_masked_image, right_eye_pos[0], right_eye_pos[1], self.fish.body.angle, lum_mask,
                                 prey_locations_array, sand_grain_locations_array)

        observation = np.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))

        return observation, full_masked_image

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
        else:
            print("No current specified.")

    def create_circular_current(self):
        unit_circle_radius = self.env_variables["unit_circle_diameter"] / 2
        circle_width = self.env_variables['current_width']
        circle_variance = self.env_variables['current_strength_variance']
        max_current_strength = self.env_variables['max_current_strength']

        arena_center = np.array([self.env_variables["arena_width"] / 2, self.env_variables["arena_height"] / 2])

        # All coordinates:
        xp, yp = np.arange(self.env_variables["arena_width"]), np.arange(self.env_variables["arena_height"])
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
        vector_field[int(self.env_variables["arena_width"] / 2), int(self.env_variables["arena_height"] / 2)] = 0

        self.impulse_vector_field = vector_field

        # plt.streamplot(xy[:, :, 0], yp[:, :, 0], vector_field[:, :, 0], vector_field[:, :, 1])
        # plt.show()

    def create_linear_current(self):
        current_width = self.env_variables['current_width'] * self.env_variables["arena_height"]
        current_variance = self.env_variables['current_strength_variance']
        max_current_strength = self.env_variables['max_current_strength']

        # Create vector field of same vectors moving in x direction
        vector = np.array([1, 0])
        vector_field = np.tile(vector, (self.env_variables["arena_width"], self.env_variables["arena_height"], 1))

        # Scale as distance from y=h/2
        centre = self.env_variables["arena_height"] / 2
        xp, yp = np.arange(self.env_variables["arena_width"]), np.arange(self.env_variables["arena_height"])
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
        """Currents act on fish only."""
        all_feature_coordinates = np.array([self.fish.body.position]).astype(int)
        # all_feature_coordinates = np.array(
        #     [self.fish.body.position] + [body.position for body in self.prey_bodies]).astype(int)
        original_orientations = np.array([self.fish.body.angle])
        # original_orientations = np.array([self.fish.body.angle] + [body.angle for body in self.prey_bodies])

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

        # for i, vector in enumerate(associated_impulse_vectors[1:]):
        #     self.prey_bodies[i].angle = np.pi
        #     self.prey_bodies[i].apply_impulse_at_local_point((vector[1], vector[0]))
        #     self.prey_bodies[i].angle = original_orientations[i + 1]

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
            xp, yp = np.arange(200, self.env_variables["arena_width"] - 200), \
                     np.arange(200, self.env_variables["arena_height"] - 200)
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
