import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale
import pymunk

from Tools.drawing_board import DrawingBoard


class BaseEnvironment:
    """A base class to represent environments, for extension to ProjectionEnvironment, VVR and Naturalistic
    environment classes."""

    def __init__(self, env_variables, draw_screen):
        self.env_variables = env_variables
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

        self.predator_bodies = []
        self.predator_shapes = []

        self.predator_body = None
        self.predator_shape = None
        self.predator_target = None

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []

        self.last_action = None  # TODO: Ensure initialisation as none doesnt have any effects.

        self.vegetation_bodies = []
        self.vegetation_shapes = []

    def reset(self):
        self.num_steps = 0
        self.fish.hungry = 0

        # TODO: Create individual methods for each removal. sand grains, prey, predators.
        for i, shp in enumerate(self.prey_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.sand_grain_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.vegetation_shapes):
            self.space.remove(shp, shp.body)

        if self.predator_shape is not None:
            self.remove_realistic_predator()

        self.prey_shapes = []
        self.prey_bodies = []

        self.predator_shapes = []
        self.predator_bodies = []

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []

        self.vegetation_bodies = []
        self.vegetation_shapes = []

    def output_frame(self, activations, internal_state, scale=0.25):
        arena = self.board.db * 255.0
        arena[0, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[self.env_variables['height'] - 1, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[:, 0, 0] = np.ones(self.env_variables['height']) * 255
        arena[:, self.env_variables['width'] - 1, 0] = np.ones(self.env_variables['height']) * 255

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

        frame = rescale(frame, scale, multichannel=True, anti_aliasing=True)
        return frame

    def draw_shapes(self):
        self.board.fish_shape(self.fish.body.position, self.env_variables['fish_mouth_size'],
                              self.env_variables['fish_head_size'], self.env_variables['fish_tail_length'],
                              self.fish.mouth.color, self.fish.mouth.color, self.fish.body.angle)  # TODO: Change second to head colour.

        if len(self.prey_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.prey_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.prey_bodies])).astype(int)
            rrs, ccs = self.board.multi_circles(px, py, self.env_variables['prey_size'])
            self.board.db[rrs, ccs] = self.prey_shapes[0].color

        if len(self.sand_grain_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.sand_grain_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.sand_grain_bodies])).astype(int)
            rrs, ccs = self.board.multi_circles(px, py, self.env_variables['sand_grain_size'])
            self.board.db[rrs, ccs] = self.sand_grain_shapes[0].color

        for i, pr in enumerate(self.predator_bodies):
            self.board.circle(pr.position, self.env_variables['predator_size'], self.predator_shapes[i].color)

        for i, pr in enumerate(self.vegetation_bodies):
            self.board.vegetation(pr.position, self.env_variables['vegetation_size'], self.vegetation_shapes[i].color)

        if self.predator_body is not None:
            self.board.circle(self.predator_body.position, self.env_variables['predator_size'],
                              self.predator_shape.color)

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
                (self.env_variables['width'] - 1, self.env_variables['height']), (self.env_variables['width'] - 1, 1),
                1),
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

    def touch_edge(self, arbiter, space, data):
        self.fish.body.velocity = (0, 0)
        if self.fish.body.angle < np.pi:
            self.fish.body.angle += np.pi
        else:
            self.fish.body.angle -= np.pi
        self.fish.body.apply_impulse_at_local_point((20, 0))

        self.fish.touched_edge = True
        return True

    def create_prey(self):
        self.prey_bodies.append(pymunk.Body(self.env_variables['prey_mass'], self.env_variables['prey_inertia']))
        self.prey_shapes.append(pymunk.Circle(self.prey_bodies[-1], self.env_variables['prey_size']))
        self.prey_shapes[-1].elasticity = 1.0
        self.prey_bodies[-1].position = (
        np.random.randint(self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                          self.env_variables['width'] - (
                                      self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'])),
        np.random.randint(self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                          self.env_variables['height'] - (
                                      self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'])))
        self.prey_shapes[-1].color = (0, 0, 1)
        self.prey_shapes[-1].collision_type = 2

        self.space.add(self.prey_bodies[-1], self.prey_shapes[-1])

    def check_paramecium_disturbance(self, prey_position):
        fish_position = self.fish.body.position
        sensing_area = [[prey_position[0] - self.env_variables['prey_sensing_distance'],
                         prey_position[0] + self.env_variables['prey_sensing_distance']],
                        [prey_position[1] - self.env_variables['prey_sensing_distance'],
                         prey_position[1] + self.env_variables['prey_sensing_distance']]]
        is_in_area = sensing_area[0][0] <= fish_position[0] <= sensing_area[0][1] and \
                     sensing_area[1][0] <= fish_position[1] <= sensing_area[1][1]
        loud_actions = [0, 1, 2]
        if is_in_area and self.last_action in loud_actions:
            print("Scared")
            return True
        else:
            return False

    def move_prey(self):
        # Not, currently, a prey isn't guaranteed to try to escape if a loud predator is near, only if it was going to
        # move anyway. Should reconsider this in the future.
        to_move = np.where(np.random.rand(len(self.prey_bodies)) < self.env_variables['prey_impulse_rate'])[0]
        angles = np.random.rand(len(to_move)) * 2 * np.pi
        for ii in range(len(to_move)):
            if self.check_paramecium_disturbance(self.prey_bodies[to_move[ii]].position):
                if self.prey_bodies[to_move[ii]].angle < (3 * np.pi) / 2:
                    self.prey_bodies[to_move[ii]].angle += np.pi / 2
                else:
                    self.prey_bodies[to_move[ii]].angle -= np.pi / 2
                self.prey_bodies[to_move[ii]].apply_impulse_at_local_point(
                    (self.env_variables['prey_escape_impulse'], 0))
            else:
                self.prey_bodies[to_move[ii]].angle = angles[ii]
                self.prey_bodies[to_move[ii]].apply_impulse_at_local_point((self.env_variables['prey_impulse'], 0))

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
        self.predator_target = fish_position  # Update so appears where fish will be in a few steps.

        self.predator_shape.color = (0, 0, 1)
        self.predator_shape.collision_type = 5

        self.space.add(self.predator_body, self.predator_shape)

    def check_predator_inside_walls(self):
        x_position, y_position = self.predator_body.position[0], self.predator_body.position[1]
        if x_position < 0:
            return True
        elif x_position > self.env_variables["width"]:
            return True
        if y_position < 0:
            return True
        elif y_position > self.env_variables["height"]:
            return True

    def move_realistic_predator(self):
        if (round(self.predator_body.position[0]), round(self.predator_body.position[1])) == (
                round(self.predator_target[0]), round(self.predator_target[1])):  # TODO: Add to method like belwow
            self.remove_realistic_predator()
            return
        if self.check_predator_inside_walls():
            self.remove_realistic_predator()
            return

        self.predator_body.angle = np.pi / 2 - np.arctan2(
            self.predator_target[0] - self.predator_body.position[0],
            self.predator_target[1] - self.predator_body.position[1])
        self.predator_body.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def remove_realistic_predator(self, arbiter=None, space=None, data=None):
        if self.predator_body is not None:
            self.space.remove(self.predator_shape, self.predator_shape.body)
            self.predator_shape = None
            self.predator_body = None
        else:
            pass

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
        self.sand_grain_shapes[-1].color = (0, 0, 1)
        self.sand_grain_shapes[-1].collision_type = 4

        self.space.add(self.sand_grain_bodies[-1], self.sand_grain_shapes[-1])

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
