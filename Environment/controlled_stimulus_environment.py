import numpy as np
import matplotlib.pyplot as plt
import pymunk

from Environment.base_environment import BaseEnvironment
from Environment.Fish.fish import Fish
from Environment.Fish.tethered_fish import TetheredFish


class ProjectionEnvironment(BaseEnvironment):
    """
    This version is made with only the fixed projection configuration in mind.
    As a result, doesnt have walls, and fish appears directly in the centre of the environment.
    For this environment, the following stimuli are available: prey, predators.
    """

    def __init__(self, env_variables, stimuli, tethered=True, draw_screen=False):
        super().__init__(env_variables, draw_screen)

        if tethered:
            self.fish = TetheredFish(self.board, env_variables, self.dark_col)
        else:
            self.fish = Fish(self.board, env_variables, self.dark_col)
        self.space.add(self.fish.body, self.fish.shape)

        # TODO: Unify in future with other stimuli
        self.prey_positions = {}
        self.predator_positions = {}

        # Whole environment measurements.
        board_height = env_variables["height"]
        board_width = env_variables["width"]

        # Wall coordinates
        self.wall_1_coordinates = [[0, 0], [0, board_height]]
        self.wall_2_coordinates = [[0, board_height], [board_width, board_height]]
        self.wall_3_coordinates = [[1, 1], [board_width,1]]
        self.wall_4_coordinates = [[board_width, 1], [board_width, board_height]]

        self.stimuli = stimuli

        self.create_walls()
        self.reset()

        self.create_positional_information(stimuli)

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_edge

    def reset(self):
        super().reset()
        self.fish.body.position = (self.env_variables['width']/2, self.env_variables['height']/2)
        self.fish.body.angle = 0
        self.fish.body.velocity = (0, 0)
        self.create_stimuli(self.stimuli)

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None):
        if frame_buffer is None:
            frame_buffer = []
        self.fish.making_capture = False
        reward = self.fish.take_action(action)

        done = False

        self.fish.hungry += (1 - self.fish.hungry)*self.env_variables['hunger_inc_tau']
        self.update_stimuli()

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.space.step(self.env_variables['phys_dt'])
            if self.fish.touched_edge:
                self.fish.touched_edge = False
            if self.show_all:
                self.board.erase()
                self.draw_shapes()
                if self.draw_screen:
                    self.board_image.set_data(self.output_frame(activations, np.array([0, 0]), scale=0.5)/255.)
                    plt.pause(0.0001)
        self.num_steps += 1
        self.board.erase()
        self.draw_shapes()

        right_eye_pos = (-np.cos(np.pi/2-self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
                         +np.sin(np.pi/2-self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (+np.cos(np.pi/2-self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
                        -np.sin(np.pi/2-self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        self.fish.left_eye.read(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        in_light = self.fish.body.position[0] > self.dark_col
        internal_state = np.array([[in_light, self.fish.hungry]])

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

        observation = np.dstack((self.readings_to_photons(self.fish.left_eye.readings),
                                 self.readings_to_photons(self.fish.right_eye.readings)))

        return observation, reward, internal_state, done, frame_buffer

    def create_stimuli(self, stimuli):
        for stimulus in stimuli:
            if "prey" in stimulus:
                self.create_prey()
            elif "predator" in stimulus:
                self.create_predator()

    def update_stimuli(self):
        # TODO: Coded very badly.
        # TODO: Fix for delayed stimuli.
        finished_prey = []
        finished_predators = []
        for i, prey in enumerate(self.prey_positions):
            try:
                self.prey_bodies[i].position = (self.prey_positions[prey][self.num_steps][0],
                                                self.prey_positions[prey][self.num_steps][1])
            except IndexError:
                self.prey_bodies.pop(i)
                self.prey_shapes.pop(i)
                finished_prey.append(prey)

        for i, predator in enumerate(self.predator_positions):
            try:
                self.predator_bodies[i].position = (self.predator_positions[predator][self.num_steps][0],
                                                    self.predator_positions[predator][self.num_steps][1])
            except IndexError:
                self.predator_bodies.pop(i)
                self.predator_shapes.pop(i)
                finished_predators.append(predator)

        for item in finished_prey:
            del self.prey_positions[item]
        for item in finished_predators:
            del self.predator_positions[item]

    def create_positional_information(self, stimuli):
        for stimulus in stimuli:
            edge_index = 0
            if "prey" in stimulus:
                self.prey_positions[stimulus] = []
                while edge_index + 1 < len(stimuli[stimulus]):
                    positions = self.interpolate_stimuli_positions(stimuli[stimulus], edge_index)
                    self.prey_positions[stimulus] = self.prey_positions[stimulus] + positions
                    edge_index += 1
            elif "predator" in stimulus:
                self.predator_positions[stimulus] = []
                while edge_index + 1 < len(stimuli[stimulus]):
                    positions = self.interpolate_stimuli_positions(stimuli[stimulus], edge_index)
                    self.predator_positions[stimulus] = self.predator_positions[stimulus] + positions
                    edge_index += 1

    @staticmethod
    def interpolate_stimuli_positions(stimulus, edge_index):
        a = stimulus[edge_index]["position"]
        b = stimulus[edge_index + 1]["position"]
        t_interval = stimulus[edge_index + 1]["step"] - stimulus[edge_index]["step"]
        dx = (b[0] - a[0])/t_interval
        dy = (b[1] - a[1])/t_interval
        interpolated_positions = [[a[0]+dx*i, a[1]+dy*i] for i in range(t_interval)]
        return interpolated_positions