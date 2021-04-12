import numpy as np
import matplotlib.pyplot as plt

from Environment.base_environment import BaseEnvironment
from Environment.Fish.fish import Fish


class NaturalisticEnvironment(BaseEnvironment):

    def __init__(self, env_variables, realistic_bouts, draw_screen=False, fish_mass=None, collisions=True):
        super().__init__(env_variables, draw_screen)

        # Create the fish class instance and add to the space.
        if fish_mass is None:
            self.fish = Fish(self.board, env_variables, self.dark_col, realistic_bouts)
        else:
            # In the event that I am producing a calibration curve for distance moved.
            self.fish = Fish(self.board, env_variables, self.dark_col, realistic_bouts, fish_mass=fish_mass)

        self.space.add(self.fish.body, self.fish.mouth, self.fish.head, self.fish.tail)

        # Create walls.
        self.create_walls()
        self.reset()

        self.col = self.space.add_collision_handler(2, 3)
        self.col.begin = self.touch_prey

        if collisions:
            self.pred_col = self.space.add_collision_handler(5, 3)
            self.pred_col.begin = self.touch_predator

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_edge

        self.edge_pred_col = self.space.add_collision_handler(1, 5)
        self.edge_pred_col.begin = self.remove_realistic_predator

        self.grain_fish_col = self.space.add_collision_handler(3, 4)
        self.grain_fish_col.begin = self.touch_grain

        # to prevent predators from knocking out prey  or static grains
        self.grain_pred_col = self.space.add_collision_handler(4, 5)
        self.grain_pred_col.begin = self.no_collision
        self.prey_pred_col = self.space.add_collision_handler(2, 5)
        self.prey_pred_col.begin = self.no_collision

    def reset(self):
        super().reset()
        self.fish.body.position = (np.random.randint(self.env_variables['fish_mouth_size'],
                                                     self.env_variables['width'] - self.env_variables[
                                                         'fish_mouth_size']),
                                   np.random.randint(self.env_variables['fish_mouth_size'],
                                                     self.env_variables['height'] - self.env_variables[
                                                         'fish_mouth_size']))
        self.fish.body.angle = np.random.random() * 2 * np.pi
        self.fish.body.velocity = (0, 0)

        for i in range(self.env_variables['prey_num']):
            self.create_prey()

        for i in range(self.env_variables['sand_grain_num']):
            self.create_sand_grain()

        for i in range(self.env_variables['vegetation_num']):
            self.create_vegetation()

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None, impulse=None):
        # TODO: Tidy up so is more readable. Do the same with comparable methods in other environment classes.
        self.prey_consumed_this_step = False
        self.last_action = action
        if frame_buffer is None:
            frame_buffer = []
        self.fish.making_capture = False

        if impulse is not None:
            # To calculate calibration curve.
            reward = self.fish.try_impulse(impulse)
        else:
            reward = self.fish.take_action(action)

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
                self.board.erase()
                self.draw_shapes()
                if self.draw_screen:
                    self.board_image.set_data(self.output_frame(activations, np.array([0, 0]), scale=0.5) / 255.)
                    plt.pause(0.0001)

        self.num_steps += 1
        self.board.erase()
        self.draw_shapes()  # TODO: Needed, but causes index error sometimes.

        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        self.fish.left_eye.read(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        # Calculate internal state
        in_light = self.fish.body.position[0] > self.dark_col
        if self.env_variables['hunger'] and self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.hungry, self.fish.stress]])
        elif self.env_variables['hunger']:
            internal_state = np.array([[in_light, self.fish.hungry]])
        elif self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.stress]])
        else:
            internal_state = np.array([[in_light]])

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

        return observation, reward, internal_state, done, frame_buffer
