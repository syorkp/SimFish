import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale
import pymunk

from Tools.drawing_board import DrawingBoard
from Environment.vis_fan import VisFan


class SimState:

    def __init__(self, env_variables, draw_screen=False):
        self.env_variables = env_variables
        self.board = DrawingBoard(self.env_variables['width'], self.env_variables['height'])
        self.draw_screen = draw_screen
        self.show_all = False
        self.num_steps = 0
        self.hungry = 0

        if self.draw_screen:
            self.board_fig, self.ax_board = plt.subplots()
            self.board_image = plt.imshow(np.zeros((self.env_variables['height'], self.env_variables['width'], 3)))
            plt.ion()
            plt.show()
        self.prey_bodies = []
        self.prey_shapes = []
        self.prey_consumed = False
        self.touched_edge = False
        self.touched_predator = False
        self.making_capture = False

        self.predator_bodies = []
        self.predator_shapes = []

        self.fish_body = None
        self.fish_shape = None

        self.dark_col = int(self.env_variables['width'] * self.env_variables['dark_light_ratio'])

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = self.env_variables['drag']

        self.create_fish(100, 100)

        self.verg_angle = self.env_variables['eyes_verg_angle'] * (np.pi / 180)
        self.retinal_field = self.env_variables['visual_field'] * (np.pi / 180)
        self.conv_state = 0

        self.left_eye = VisFan(self.board, self.verg_angle, self.retinal_field, True,
                               self.env_variables['num_photoreceptors'], self.env_variables['min_vis_dist'],
                               self.env_variables['max_vis_dist'], self.env_variables['dark_gain'],
                               self.env_variables['light_gain'], self.env_variables['bkg_scatter'], self.dark_col)

        self.right_eye = VisFan(self.board, self.verg_angle, self.retinal_field, False,
                                self.env_variables['num_photoreceptors'], self.env_variables['min_vis_dist'],
                                self.env_variables['max_vis_dist'], self.env_variables['dark_gain'],
                                self.env_variables['light_gain'], self.env_variables['bkg_scatter'], self.dark_col)

        # Create walls.
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

        self.reset()

        self.col = self.space.add_collision_handler(2, 3)
        self.col.begin = self.touch_prey

        self.pred_col = self.space.add_collision_handler(5, 3)
        self.pred_col.begin = self.touch_predator

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_edge

    def reset(self):
        self.num_steps = 0
        self.hungry = 0
        self.fish_body.position = (np.random.randint(self.env_variables['fish_size'], self.env_variables['width'] - self.env_variables['fish_size']),
                                   np.random.randint(self.env_variables['fish_size'], self.env_variables['height'] - self.env_variables['fish_size']))
        self.fish_body.angle = np.random.random()*2*np.pi
        self.fish_body.velocity = (0, 0)
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

    def readings_to_photons(self, readings):
        photons = np.random.poisson(readings * self.env_variables['photon_ratio'])
        if self.env_variables['read_noise_sigma'] > 0:
            noise = np.random.randn(readings.shape[0], readings.shape[1]) * self.env_variables['read_noise_sigma']
            photons += noise.astype(int)
        return photons

    def touch_prey(self, arbiter, space, data):
        if self.making_capture:
            for i, shp in enumerate(self.prey_shapes):
                if shp == arbiter.shapes[0]:
                    space.remove(shp, shp.body)
                    self.prey_shapes.remove(shp)
                    self.prey_bodies.remove(shp.body)

            self.prey_consumed = True
            return False
        else:
            return True

    def touch_predator(self, arbiter, space, data):
        if self.num_steps > self.env_variables['immunity_steps']:
            self.touched_predator = True
            return False
        else:
            return True

    def touch_edge(self, arbiter, space, data):
        self.fish_body.velocity = (0, 0)
        if self.fish_body.angle < np.pi:
            self.fish_body.angle += np.pi
        else:
            self.fish_body.angle -= np.pi
        self.fish_body.apply_impulse_at_local_point((20, 0))

        self.touched_edge = True
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
            dist_to_fish = np.sqrt((pr.position[0] - self.fish_body.position[0])**2 + (pr.position[1] - self.fish_body.position[1])**2)

            if dist_to_fish < self.env_variables['predator_sensing_dist']:
                pr.angle = np.pi/2 - np.arctan2(self.fish_body.position[0] - pr.position[0], self.fish_body.position[1] - pr.position[1])
                pr.apply_impulse_at_local_point((self.env_variables['predator_chase_impulse'], 0))

            elif np.random.rand(1) < self.env_variables['predator_impulse_rate']:
                pr.angle = np.random.rand(1)*2*np.pi
                pr.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def create_fish(self, x, y):
        inertia = pymunk.moment_for_circle(self.env_variables['fish_mass'], 0, self.env_variables['fish_size'], (0, 0))
        self.fish_body = pymunk.Body(1, inertia)
        self.fish_shape = pymunk.Circle(self.fish_body, self.env_variables['fish_size'])
        self.fish_shape.color = (0, 1, 0)
        self.fish_shape.elasticity = 1.0
        self.fish_shape.collision_type = 3
        self.space.add(self.fish_body, self.fish_shape)

    def output_frame(self, activations, internal_state, scale=0.25):
        arena = self.board.db*255.0
        arena[0, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[self.env_variables['height'] - 1, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[:, 0, 0] = np.ones(self.env_variables['height']) * 255
        arena[:, self.env_variables['width'] - 1, 0] = np.ones(self.env_variables['height']) * 255

        left_photons = self.readings_to_photons(self.left_eye.readings)
        right_photons = self.readings_to_photons(self.right_eye.readings)
        left_eye = resize(np.reshape(left_photons, (1, len(self.left_eye.vis_angles), 3)) * (255 / self.env_variables['photon_ratio']), (20, self.env_variables['width'] / 2 - 50))
        right_eye = resize(np.reshape(right_photons, (1, len(self.right_eye.vis_angles), 3)) * (255 / self.env_variables['photon_ratio']), (20, self.env_variables['width'] / 2 - 50))
        eyes = np.hstack((left_eye, np.zeros((20, 100, 3)), right_eye))
        eyes[eyes < 0] = 0
        eyes[eyes > 255] = 255

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
                pos = (activations[ac] - adr[0]) / (adr[1]-adr[0])

                pos[pos < 0] = 0
                pos[pos > 1] = 1

                this_ac[:, :, 0] = resize(pos, (20, self.env_variables['width'])) * 255
                this_ac[:, :, 1] = resize(pos, (20, self.env_variables['width'])) * 255
                this_ac[:, :, 2] = resize(pos, (20, self.env_variables['width'])) * 255

                frame = np.vstack((frame, np.zeros((20, self.env_variables['width'], 3)), this_ac))

        frame = rescale(frame, scale, multichannel=True, anti_aliasing=True)
        return frame

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None):
        if frame_buffer is None:
            frame_buffer = []
        self.making_capture = False
        reward = None
        if action == 0:  # Swim forward
            reward = -self.env_variables['forward_swim_cost']
            self.fish_body.apply_impulse_at_local_point((self.env_variables['forward_swim_impulse'], 0))
            self.fish_shape.color = (0, 1, 0)
        elif action == 1:  # Turn right
            reward = -self.env_variables['routine_turn_cost']
            self.fish_body.angle += self.env_variables['routine_turn_dir_change']
            self.fish_body.apply_impulse_at_local_point((self.env_variables['routine_turn_impulse'], 0))
            self.fish_shape.color = (0, 1, 0)
        elif action == 2:   # Turn left
            reward = -self.env_variables['routine_turn_cost']
            self.fish_body.angle -= self.env_variables['routine_turn_dir_change']
            self.fish_body.apply_impulse_at_local_point((self.env_variables['routine_turn_impulse'], 0))
            self.fish_shape.color = (0, 1, 0)
        elif action == 3:   # Capture
            reward = -self.env_variables['capture_swim_cost']
            self.fish_body.apply_impulse_at_local_point((self.env_variables['capture_swim_impulse'], 0))
            self.fish_shape.color = [1, 0, 1]
            self.making_capture = True
        elif action == 4:  # j turn right
            reward = -self.env_variables['j_turn_cost']
            self.fish_body.angle += self.env_variables['j_turn_dir_change']
            self.fish_body.apply_impulse_at_local_point((self.env_variables['j_turn_impulse'], 0))
            self.fish_shape.color = [1, 1, 1]
        elif action == 5:  # j turn left
            reward = -self.env_variables['j_turn_cost']
            self.fish_body.angle -= self.env_variables['j_turn_dir_change']
            self.fish_body.apply_impulse_at_local_point((self.env_variables['j_turn_impulse'], 0))
            self.fish_shape.color = [1, 1, 1]
        elif action == 6:   # do nothing:
            reward = -self.env_variables['rest_cost']
        else:
            print("Invalid action given")

        # elif action == 6: #converge eyes
        #     self.verg_angle = 77 * (np.pi / 180)
        #     self.left_eye.update_angles(self.verg_angle, self.retinal_field, True)
        #     self.right_eye.update_angles(self.verg_angle, self.retinal_field, False)
        #     self.conv_state = 1

        # elif action == 7: #diverge eyes
        #     self.verg_angle = 25 * (np.pi / 180)
        #     self.left_eye.update_angles(self.verg_angle, self.retinal_field, True)
        #     self.right_eye.update_angles(self.verg_angle, self.retinal_field, False)
        #     self.conv_state = 0

        done = False
        self.hungry += (1 - self.hungry)*self.env_variables['hunger_inc_tau']

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.move_prey()
            self.move_predator()

            self.space.step(self.env_variables['phys_dt'])
            if self.prey_consumed:
                reward += self.env_variables['capture_basic_reward'] * self.hungry
                self.hungry *= self.env_variables['hunger_dec_tau']
                if len(self.prey_shapes) == 0:
                    done = True
                self.prey_consumed = False
            if self.touched_edge:
                self.touched_edge = False
            if self.touched_predator:
                reward -= self.env_variables['predator_cost']
                done = True
                self.touched_predator = False

            if self.show_all:
                self.board.erase()
                self.draw_shapes()
                if self.draw_screen:
                    self.board_image.set_data(self.output_frame(activations, np.array([0, 0]), scale=0.5)/255.)
                    plt.pause(0.0001)
        self.num_steps += 1
        self.board.erase()
        self.draw_shapes()
        right_eye_pos = (-np.cos(np.pi/2-self.fish_body.angle) * self.env_variables['eyes_biasx'] + self.fish_body.position[0],
                         +np.sin(np.pi/2-self.fish_body.angle) * self.env_variables['eyes_biasx'] + self.fish_body.position[1])
        left_eye_pos = (+np.cos(np.pi/2-self.fish_body.angle) * self.env_variables['eyes_biasx'] + self.fish_body.position[0],
                        -np.sin(np.pi/2-self.fish_body.angle) * self.env_variables['eyes_biasx'] + self.fish_body.position[1])

        self.left_eye.read(left_eye_pos[0], left_eye_pos[1], self.fish_body.angle)
        self.right_eye.read(right_eye_pos[0], right_eye_pos[1], self.fish_body.angle)

        # Calculate internal state
        in_light = self.fish_body.position[0] > self.dark_col
        internal_state = np.array([[in_light, self.hungry]])

        if save_frames or self.draw_screen:
            self.board.erase(bkg=self.env_variables['bkg_scatter'])
            self.draw_shapes()
            self.board.apply_light(self.dark_col, 0.7, 1)
            self.left_eye.show_points(left_eye_pos[0], left_eye_pos[1], self.fish_body.angle)
            self.right_eye.show_points(right_eye_pos[0], right_eye_pos[1], self.fish_body.angle)
            if save_frames:
                frame_buffer.append(self.output_frame(activations, internal_state, scale=0.25))
            if self.draw_screen:
                self.board_image.set_data(self.output_frame(activations, internal_state, scale=0.5)/255.)
                plt.pause(0.000001)

        observation = np.dstack((self.readings_to_photons(self.left_eye.readings), self.readings_to_photons(self.right_eye.readings)))

        return observation, reward, internal_state, done, frame_buffer

    def draw_shapes(self):
        self.board.circle(self.fish_body.position, self.env_variables['fish_size'], self.fish_shape.color)

        if len(self.prey_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.prey_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.prey_bodies])).astype(int)
            rrs, ccs = self.board.multi_circles(px, py, self.env_variables['prey_size'])
            self.board.db[rrs, ccs] = self.prey_shapes[0].color

        for i, pr in enumerate(self.predator_bodies):
            self.board.circle(pr.position, self.env_variables['predator_size'], self.predator_shapes[i].color)
