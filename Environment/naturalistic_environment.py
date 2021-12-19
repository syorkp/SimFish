import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from Environment.base_environment import BaseEnvironment
from Environment.Fish.fish import Fish


class NaturalisticEnvironment(BaseEnvironment):

    def __init__(self, env_variables, realistic_bouts, new_simulation, using_gpu, draw_screen=False, fish_mass=None, collisions=True):
        super().__init__(env_variables, draw_screen, new_simulation, using_gpu)

        self.using_gpu = using_gpu

        if using_gpu:
            self.chosen_math_library = cp
        else:
            self.chosen_math_library = np

    def reset(self):
        super().reset()
        self.fish.body.position = (np.random.randint(self.env_variables['fish_mouth_size'] + 40,
                                                     self.env_variables['width'] - (self.env_variables[
                                                         'fish_mouth_size']+40)),
                                   np.random.randint(self.env_variables['fish_mouth_size'] + 40,
                                                     self.env_variables['height'] - (self.env_variables[
                                                         'fish_mouth_size']+40)))
        self.fish.body.angle = np.random.random() * 2 * np.pi
        self.fish.body.velocity = (0, 0)
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
                for cloud in range(self.env_variables["prey_cloud_num"])]
            self.build_prey_cloud_walls()

        for i in range(self.env_variables['prey_num']):
            self.create_prey()

        for i in range(self.env_variables['sand_grain_num']):
            self.create_sand_grain()

        for i in range(self.env_variables['vegetation_num']):
            self.create_vegetation()

    def show_new_channel_sectors(self, left_eye_pos, right_eye_pos):
        left_sectors, right_sectors = self.fish.get_all_sectors([left_eye_pos[0], left_eye_pos[1]], [right_eye_pos[0], right_eye_pos[1]], self.fish.body.angle)
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

    def simulation_step(self, action, save_frames, frame_buffer, activations, impulse):

        self.prey_consumed_this_step = False
        self.last_action = action
        if frame_buffer is None:
            frame_buffer = []

        if impulse is not None:
            # To calculate calibration curve.
            reward = self.fish.try_impulse(impulse)
        else:
            reward = self.fish.take_action(action)

        # Add policy helper reward to encourage proximity to prey.
        for ii in range(len(self.prey_bodies)):
            if self.check_proximity(self.prey_bodies[ii].position, self.env_variables['reward_distance']):
                reward += self.env_variables['proximity_reward']

        done = False

        self.fish.hungry += (1 - self.fish.hungry) * self.env_variables['hunger_inc_tau']
        self.fish.stress = self.fish.stress * self.env_variables['stress_compound']
        if self.predator_body is not None:
            self.fish.stress += 0.5

        # TODO: add below to function for clarity.
        if self.predator_shape is None and np.random.rand(1) < self.env_variables["probability_of_predator"] and \
                self.num_steps > self.env_variables['immunity_steps'] and not self.check_fish_near_vegetation():
            self.create_realistic_predator()

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.move_prey()
            self.displace_sand_grains()
            if self.predator_body is not None:
                self.move_realistic_predator()

            self.space.step(self.env_variables['phys_dt'])
            if self.fish.prey_consumed:
                reward += self.env_variables['capture_basic_reward'] * self.fish.hungry
                self.fish.hungry *= self.env_variables['hunger_dec_tau']
                if len(self.prey_shapes) == 0:
                    done = True
                self.fish.prey_consumed = False
            if self.fish.touched_edge:
                self.fish.touched_edge = False
            if self.fish.touched_predator:
                reward -= self.env_variables['predator_cost']
                done = True
                self.fish.touched_predator = False

            if self.show_all:
                # TODO: ENV CHANGE
                self.board.erase()
                self.draw_shapes()
                if self.draw_screen:
                    self.board_image.set_data(self.output_frame(activations, np.array([0, 0]), scale=0.5) / 255.)
                    plt.pause(0.0001)

        self.num_steps += 1
        self.board.erase()
        self.draw_shapes()

        # Calculate internal state TODO: Moved this above first of visual input function
        in_light = self.fish.body.position[0] > self.dark_col
        if self.env_variables['hunger'] and self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.hungry, self.fish.stress]])
        elif self.env_variables['hunger']:
            internal_state = np.array([[in_light, self.fish.hungry]])
        elif self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.stress]])
        else:
            internal_state = np.array([[in_light]])

        if self.new_simulation:
            observation, frame_buffer = self.resolve_visual_input_new(save_frames, activations, internal_state, frame_buffer)
        else:
            observation, frame_buffer = self.resolve_visual_input(save_frames, activations, internal_state, frame_buffer)

        return observation, reward, internal_state, done, frame_buffer

    def resolve_visual_input_new(self, save_frames, activations, internal_state, frame_buffer):
        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        # self.show_new_channel_sectors(left_eye_pos, right_eye_pos)
        if self.predator_body is not None:
            predator_bodies = [self.predator_body.position]
        else:
            predator_bodies = []
        full_masked_image = self.board.get_masked_pixels(self.fish.body.position,
                                                         [i.position for i in self.prey_bodies],
                                                         predator_bodies
                                                         )
        # plt.imshow(full_masked_image * 100)
        # plt.show()

        self.fish.left_eye.read(full_masked_image, left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(full_masked_image, right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        if save_frames or self.draw_screen:
            self.board.erase(bkg=self.env_variables['bkg_scatter'])
            self.draw_shapes()
            self.board.apply_light(self.dark_col, 0.7, 1)
            self.fish.left_eye.show_points(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
            self.fish.right_eye.show_points(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)
            if save_frames:
                frame_buffer.append(self.output_frame(activations, internal_state, scale=0.25))
            if self.draw_screen:
                self.board_image.set_data(self.output_frame(activations, internal_state, scale=0.5) / 255.)
                plt.pause(0.000001)

        observation = self.chosen_math_library.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))
        # self.plot_observation(observation)

        if self.using_gpu:
            return observation.get(), frame_buffer
        else:
            return observation, frame_buffer

    def plot_observation(self, observation):
        if self.using_gpu:
            observation = observation.get()
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
        # TODO: ENV CHANGE Maybe have additional visual input function?
        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        self.fish.left_eye.read(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        if save_frames or self.draw_screen:
            self.board.erase(bkg=self.env_variables['bkg_scatter'])
            self.draw_shapes()
            self.board.apply_light(self.dark_col, 0.7, 1)
            self.fish.left_eye.show_points(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
            self.fish.right_eye.show_points(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)
            if save_frames:
                frame_buffer.append(self.output_frame(activations, internal_state, scale=0.25))
            if self.draw_screen:
                self.board_image.set_data(self.output_frame(activations, internal_state, scale=0.5) / 255.)
                plt.pause(0.000001)

        observation = np.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))
        return observation
