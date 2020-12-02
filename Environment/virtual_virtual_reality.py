import numpy as np
import matplotlib.pyplot as plt
import pymunk

from Environment.base_environment import BaseEnvironment
from Environment.fish import Fish


class VirtualVirtualReality(BaseEnvironment):

    def __init__(self, env_variables, draw_screen=False):
        super().__init__(env_variables, draw_screen)

        # TODO: Add in fixed option later

        self.fish = Fish(self.board, env_variables, self.dark_col)
        self.space.add(self.fish.body, self.fish.shape)

        # Whole environment measurements. TODO: Replace all in program uses of env_variables reading with these.
        board_height = env_variables["height"]
        board_width = env_variables["width"]

        # The projections, which span the edges of the board. TODO: Check creates projections in correct place.
        # self.projection_1_coordinates = [[0, 0], [0, board_height]]
        # self.projection_2_coordinates = [[0, board_height], [board_width, board_height]]
        # self.projection_3_coordinates = [[board_width, board_height], [board_width, 0]]
        # self.projection_4_coordinates = [[0, 0], [board_width, 0]]  # Decide if want this.
        self.projection_1_coordinates = [[0, 1], [0, board_height]]
        self.projection_2_coordinates = [[1, board_height], [board_width, board_height]]
        self.projection_3_coordinates = [[board_width - 1, board_height], [board_width -1, 1]]
        self.projection_4_coordinates = [[1, 1], [board_width, 1]]  # Decide if want this.

        # Wall coordinates
        self.wall_1_coordinates = [[0.1 * board_width, 0.1 * board_height], [0.1 * board_width, 0.9 * board_height]]
        self.wall_2_coordinates = [[0.1 * board_width, 0.9 * board_height], [0.9 * board_width, 0.9 * board_height]]
        self.wall_3_coordinates = [[0.1 * board_width, 0.1 * board_height], [0.9 * board_width, 0.1 * board_height]]
        self.wall_4_coordinates = [[0.9 * board_width, 0.1 * board_height], [0.9 * board_width, 0.9 * board_height]]

        self.create_walls()
        self.reset()

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_edge

    def reset(self):
        # TODO: Split parts of this into new methods in the BaseEnvironment class, such as reset_fish()
        self.num_steps = 0
        self.fish.hungry = 0

        self.fish.body.position = (np.random.randint(self.env_variables['fish_size'], self.env_variables['width'] - self.env_variables['fish_size']),
                                   np.random.randint(self.env_variables['fish_size'], self.env_variables['height'] - self.env_variables['fish_size']))
        self.fish.body.angle = np.random.random()*2*np.pi
        self.fish.body.velocity = (0, 0)

    def create_walls(self):
        # TODO: At present, could easily belong in the base class.
        # TODO: Walls need to allow for the projection to exist beyond them
        # TODO: Walls need to be transparent
        # static = [
        #     pymunk.Segment(
        #         self.space.static_body,
        #         (0, 1), (0, self.wall_h), 1),
        #     pymunk.Segment(
        #         self.space.static_body,
        #         (1, self.wall_h), (self.wall_w, self.wall_h), 1),
        #     pymunk.Segment(
        #         self.space.static_body,
        #         (self.wall_w - 1, self.wall_h), (self.wall_w - 1, 1), 1),
        #     pymunk.Segment(
        #         self.space.static_body,
        #         (1, 1), (self.wall_w, 1), 1)
        # ]
        static = [
            pymunk.Segment(
                self.space.static_body,
                self.wall_1_coordinates[0], self.wall_1_coordinates[1], 1),
            pymunk.Segment(
                self.space.static_body,
                self.wall_2_coordinates[0], self.wall_2_coordinates[1], 1),
            pymunk.Segment(
                self.space.static_body,
                self.wall_3_coordinates[0], self.wall_3_coordinates[1], 1),
            pymunk.Segment(
                self.space.static_body,
                self.wall_4_coordinates[0], self.wall_4_coordinates[1], 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            # s.color = (1, 1, 1)  # TODO: Test is translucent
        self.space.add(static)

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None):
        if frame_buffer is None:
            frame_buffer = []
        self.fish.making_capture = False
        reward = self.fish.take_action(action)

        done = False

        self.fish.hungry += (1 - self.fish.hungry)*self.env_variables['hunger_inc_tau']

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.project()  # Update the projection views

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

    def project(self, projection_array=None):
        projections = [
            pymunk.Segment(self.space.static_body, self.projection_1_coordinates[0], self.projection_1_coordinates[1],
                           1),
            pymunk.Segment(self.space.static_body, self.projection_2_coordinates[0], self.projection_2_coordinates[1],
                           1),
            pymunk.Segment(self.space.static_body, self.projection_3_coordinates[0], self.projection_3_coordinates[1],
                           1),
            pymunk.Segment(self.space.static_body, self.projection_4_coordinates[0], self.projection_4_coordinates[1],
                           1),
        ] # TODO: May need to add 1 on as was done in wall objects.

        for s in projections:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = (0, 0, 1)  # TODO: Test is visible
        self.space.add(projections)
        # In future, could make it so that it takes a predator shape and then flattens it.
        # TODO: Research how to create a visible shape in Pymunk
        # TODO: Start with just a blue line

    def draw_shapes(self):
        self.board.circle(self.fish.body.position, self.env_variables['fish_size'], self.fish.shape.color)


