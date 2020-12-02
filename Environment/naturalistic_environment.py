import numpy as np
import matplotlib.pyplot as plt
import pymunk

from Environment.base_environment import BaseEnvironment
from Environment.fish import Fish


class NaturalisticEnvironment(BaseEnvironment):

    # TODO: Consider adding touch and move methods to base - will require placeholders there.

    def __init__(self, env_variables, draw_screen=False):
        super().__init__(env_variables, draw_screen)

        self.prey_bodies = []
        self.prey_shapes = []

        self.predator_bodies = []
        self.predator_shapes = []

        # Create the fish class instance and add to the space.
        self.fish = Fish(self.board, env_variables, self.dark_col)
        self.space.add(self.fish.body, self.fish.shape)

        # Create walls.
        self.create_walls()
        self.reset()

        self.col = self.space.add_collision_handler(2, 3)
        self.col.begin = self.touch_prey

        self.pred_col = self.space.add_collision_handler(5, 3)
        self.pred_col.begin = self.touch_predator

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_edge

    def reset(self):
        # TODO: Add overlap for different environments to base method.
        self.num_steps = 0
        self.fish.hungry = 0

        self.fish.body.position = (np.random.randint(self.env_variables['fish_size'], self.env_variables['width'] - self.env_variables['fish_size']),
                                   np.random.randint(self.env_variables['fish_size'], self.env_variables['height'] - self.env_variables['fish_size']))
        self.fish.body.angle = np.random.random()*2*np.pi
        self.fish.body.velocity = (0, 0)
        for i, shp in enumerate(self.prey_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.predator_shapes):
            self.space.remove(shp, shp.body)

        self.prey_shapes = []
        self.prey_bodies = []
        self.predator_shapes = []
        self.predator_bodies = []

        for i in range(self.env_variables['prey_num']):
            self.create_prey()

        for i in range(self.env_variables['predator_num']):
            self.create_predator()

    def create_walls(self):
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, self.env_variables['height']), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, self.env_variables['height']), (self.env_variables['width'], self.env_variables['height']), 1),
            pymunk.Segment(
                self.space.static_body,
                (self.env_variables['width'] - 1, self.env_variables['height']), (self.env_variables['width'] - 1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (self.env_variables['width'], 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = (1, 0, 0)
        self.space.add(static)

    def touch_prey(self, arbiter, space, data):
        if self.fish.making_capture:
            for i, shp in enumerate(self.prey_shapes):
                if shp == arbiter.shapes[0]:
                    space.remove(shp, shp.body)
                    self.prey_shapes.remove(shp)
                    self.prey_bodies.remove(shp.body)

            self.fish.prey_consumed = True
            return False
        else:
            return True

    def touch_predator(self, arbiter, space, data):
        if self.num_steps > self.env_variables['immunity_steps']:
            self.fish.touched_predator = True
            return False
        else:
            return True

    def create_prey(self):
        self.prey_bodies.append(pymunk.Body(self.env_variables['prey_mass'], self.env_variables['prey_inertia']))
        self.prey_shapes.append(pymunk.Circle(self.prey_bodies[-1], self.env_variables['prey_size']))
        self.prey_shapes[-1].elasticity = 1.0
        self.prey_bodies[-1].position = (np.random.randint(self.env_variables['prey_size'] + self.env_variables['fish_size'],
                                                           self.env_variables['width'] - (self.env_variables['prey_size'] + self.env_variables['fish_size'])),
                                         np.random.randint(self.env_variables['prey_size'] + self.env_variables['fish_size'],
                                                           self.env_variables['height'] - (self.env_variables['prey_size'] + self.env_variables['fish_size'])))
        self.prey_shapes[-1].color = (0, 0, 1)
        self.prey_shapes[-1].collision_type = 2

        self.space.add(self.prey_bodies[-1], self.prey_shapes[-1])

    def create_predator(self):
        self.predator_bodies.append(pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia']))
        self.predator_shapes.append(pymunk.Circle(self.predator_bodies[-1], self.env_variables['predator_size']))
        self.predator_shapes[-1].elasticity = 1.0
        self.predator_bodies[-1].position = (np.random.randint(self.env_variables['predator_size'] + self.env_variables['fish_size'],
                                                               self.env_variables['width'] - (self.env_variables['predator_size'] + self.env_variables['fish_size'])),
                                             np.random.randint(self.env_variables['predator_size'] + self.env_variables['fish_size'],
                                                               self.env_variables['height'] - (self.env_variables['predator_size'] + self.env_variables['fish_size'])))
        self.predator_shapes[-1].color = (0, 0, 1)
        self.predator_shapes[-1].collision_type = 5

        self.space.add(self.predator_bodies[-1], self.predator_shapes[-1])

    def move_prey(self):
        to_move = np.where(np.random.rand(len(self.prey_bodies)) < self.env_variables['prey_impulse_rate'])[0]
        angles = np.random.rand(len(to_move))*2*np.pi
        for ii in range(len(to_move)):
            self.prey_bodies[to_move[ii]].angle = angles[ii]
            self.prey_bodies[to_move[ii]].apply_impulse_at_local_point((self.env_variables['prey_impulse'], 0))

    def move_predator(self):
        for pr in self.predator_bodies:
            dist_to_fish = np.sqrt((pr.position[0] - self.fish.body.position[0])**2 + (pr.position[1] - self.fish.body.position[1])**2)

            if dist_to_fish < self.env_variables['predator_sensing_dist']:
                pr.angle = np.pi/2 - np.arctan2(self.fish.body.position[0] - pr.position[0], self.fish.body.position[1] - pr.position[1])
                pr.apply_impulse_at_local_point((self.env_variables['predator_chase_impulse'], 0))

            elif np.random.rand(1) < self.env_variables['predator_impulse_rate']:
                pr.angle = np.random.rand(1)*2*np.pi
                pr.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None):
        if frame_buffer is None:
            frame_buffer = []
        self.fish.making_capture = False
        reward = self.fish.take_action(action)

        done = False

        self.fish.hungry += (1 - self.fish.hungry)*self.env_variables['hunger_inc_tau']

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.move_prey()
            self.move_predator()

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

        # Calculate internal state
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
                self.board_image.set_data(self.output_frame(activations, internal_state, scale=0.5)/255.)
                plt.pause(0.000001)

        observation = np.dstack((self.readings_to_photons(self.fish.left_eye.readings), self.readings_to_photons(self.fish.right_eye.readings)))

        return observation, reward, internal_state, done, frame_buffer

    def draw_shapes(self):
        self.board.circle(self.fish.body.position, self.env_variables['fish_size'], self.fish.shape.color)

        if len(self.prey_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.prey_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.prey_bodies])).astype(int)
            rrs, ccs = self.board.multi_circles(px, py, self.env_variables['prey_size'])
            self.board.db[rrs, ccs] = self.prey_shapes[0].color

        for i, pr in enumerate(self.predator_bodies):
            self.board.circle(pr.position, self.env_variables['predator_size'], self.predator_shapes[i].color)


