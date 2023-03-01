import numpy as np
import matplotlib.pyplot as plt
import copy

import pymunk

from Environment.Action_Space.draw_angle_dist_new import draw_angle_dist_new as draw_angle_dist_narrowed
from Environment.Action_Space.draw_angle_dist import convert_action_to_bout_id



class TestEnvironment:

    def __init__(self, fraction_capture_possible=0.2, permitted_angular_deviation=np.pi/6, predator_impulse=25):
        self.touched_predator = False

        self.env_variables = {
            'phys_dt': 0.2,  # physics time step
            'drag': 0.7,  # water drag

            'fish_mass': 140.,
            'fish_mouth_radius': 4.,  # FINAL VALUE - 0.2mm diameter, so 1.
            'fish_head_radius': 2.5,  # Old - 10
            'fish_tail_length': 41.5,  # Old: 70
            'eyes_verg_angle': 77.,  # in deg

            'prey_mass': 1.,
            'prey_inertia': 40.,
            'prey_size': 1,  # FINAL VALUE - 0.2mm diameter, so 1.
            'prey_max_turning_angle': 0.25,
            'prey_sensing_distance': 20,
            'prey_jump': True,
            'prey_fluid_displacement': True,
            'prey_cloud_num': 16,

            # Prey movement
            'p_slow': 1.0,
            'p_fast': 0.0,
            'p_escape': 0.5,
            'p_switch': 0.01,  # Corresponds to 1/average duration of movement type.
            'p_reorient': 0.04,
            'slow_impulse_paramecia': 0.0035, # Actual values should be 0.0035,  # Impulse to generate 0.5mms-1 for given prey mass
            'fast_impulse_paramecia': 0.007, # Actual values should be 0.07,  # Impulse to generate 1.0mms-1 for given prey mass
            'jump_impulse_paramecia': 0.1, # Actual values should be 0.7,  # Impulse to generate 10.0mms-1 for given prey mass

            'displacement_scaling_factor': 0.018,
            # Multiplied by previous impulse size to cause displacement of nearby features.
            'known_max_fish_i': 100,

            'predator_mass': 200.,
            'predator_inertia': 0.0001,
            'predator_size': 32,
            'predator_impulse': predator_impulse,

            "max_predator_attacks": 5,
            "further_attack_probability": 0.4,
            "max_predator_attack_range": 2000,
            "max_predator_reorient_distance": 400,
            "predator_presence_duration_steps": 100,
            "predator_first_attack_loom": False,
            "initial_predator_size": 20,
            "final_predator_size": 200,
        }

        # Fish params
        inertia = pymunk.moment_for_circle(self.env_variables['fish_mass'], 0, self.env_variables['fish_head_radius'],
                                           (0, 0))
        self.body = pymunk.Body(1, inertia)
        # Mouth
        self.mouth = pymunk.Circle(self.body, self.env_variables['fish_mouth_radius'], offset=(0, 0))
        self.mouth.color = (0, 1, 0)
        self.mouth.elasticity = 1.0
        self.mouth.collision_type = 3

        # Head
        self.head = pymunk.Circle(self.body, self.env_variables['fish_head_radius'],
                                  offset=(-self.env_variables['fish_head_radius'], 0))
        self.head.color = (0, 1, 0)
        self.head.elasticity = 1.0
        self.head.collision_type = 6

        # # Tail
        tail_coordinates = ((-self.env_variables['fish_head_radius'], 0),
                            (-self.env_variables['fish_head_radius'], - self.env_variables['fish_head_radius']),
                            (-self.env_variables['fish_head_radius'] - self.env_variables['fish_tail_length'], 0),
                            (-self.env_variables['fish_head_radius'], self.env_variables['fish_head_radius']))
        self.tail = pymunk.Poly(self.body, tail_coordinates)
        self.tail.color = (0, 1, 0)
        self.tail.elasticity = 1.0
        self.tail.collision_type = 6

        self.body.position = (500, 500)
        self.body.angle = np.random.random() * 2 * np.pi
        self.body.velocity = (0, 0)

        self.prey_bodies = []
        self.prey_shapes = []
        self.paramecia_gaits = []

        self.predator_bodies = []
        self.predator_shapes = []

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = self.env_variables['drag']
        self.space.add(self.body, self.mouth, self.head, self.tail)

        self.col = self.space.add_collision_handler(2, 3)
        self.col.begin = self.touch_prey

        self.prev_action_impulse = 0

        # Capture dynamics
        self.fraction_capture_possible, self.permitted_angular_deviation = fraction_capture_possible, permitted_angular_deviation

        self.capture_fraction = int(
            100 * fraction_capture_possible)
        self.capture_start = 1 # int((9 - self.capture_fraction) / 2)
        self.capture_end = self.capture_start + self.capture_fraction

        self.prey_consumed_this_step = False

        # Predator
        self.predator_body = None
        self.predator_shape = None
        self.predator_target = None

        # New complex predators
        self.predator_location = None
        self.remaining_predator_attacks = None
        self.total_predator_steps = None
        self.new_attack_due = False

        self.pred_col = self.space.add_collision_handler(5, 3)
        self.pred_col.begin = self.touch_predator

        self.pred_col2 = self.space.add_collision_handler(5, 6)
        self.pred_col2.begin = self.touch_predator
        self.micro_step = 0

    def touch_prey(self, arbiter, space, data):
        valid_capture = False
        if self.capture_possible:
            for i, shp in enumerate(self.prey_shapes):
                if shp == arbiter.shapes[0]:
                    # Check if angles line up.
                    prey_position = self.prey_bodies[i].position
                    fish_position = self.body.position
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
                    fish_orientation = (self.body.angle % (2 * np.pi))

                    # Normalise so both in same reference frame
                    deviation = abs(fish_orientation - angle)
                    if deviation > np.pi:
                        # Need to account for cases where one angle is very high, while other is very low, as these
                        # angles can be close together. Can do this by summing angles and subtracting from 2 pi.
                        deviation -= (2 * np.pi)
                        deviation = abs(deviation)
                    if deviation < self.permitted_angular_deviation:
                        self.latest_incidence = deviation
                        # print("Successful capture \n")
                        valid_capture = True
                        space.remove(shp, shp.body)
                        self.prey_shapes.remove(shp)
                        self.prey_bodies.remove(shp.body)
                        del self.paramecia_gaits[i]
                    else:
                        pass
                        # print("Failed capture \n")
                        # print(f"""Prey position: {prey_position}
                        # Fish position: {fish_position}
                        # Fish orientation: {fish_orientation}
                        # Computed orientation: {angle}
                        # """)

            if valid_capture:
                self.prey_consumed = True
                self.prey_consumed_this_step = True

            return False
        else:
            return True

    def create_prey(self, prey_position):
        self.prey_bodies.append(pymunk.Body(self.env_variables['prey_mass'], self.env_variables['prey_inertia']))
        self.prey_shapes.append(pymunk.Circle(self.prey_bodies[-1], self.env_variables['prey_size']))
        self.prey_shapes[-1].elasticity = 1.0
        self.prey_bodies[-1].position = prey_position
        self.prey_shapes[-1].color = (0, 0, 1)
        self.prey_shapes[-1].collision_type = 2
        # self.prey_shapes[-1].filter = pymunk.ShapeFilter(
        #     mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.prey_bodies[-1], self.prey_shapes[-1])

        # New prey motion TODO: Check doesnt mess with base version.
        # self.paramecia_gaits.append(
        #     np.random.choice([0, 1, 2], 1, p=[1 - (self.env_variables["p_fast"] + self.env_variables["p_slow"]),
        #                                       self.env_variables["p_slow"],
        #                                       self.env_variables["p_fast"]])[0])
        # New prey motion
        self.paramecia_gaits.append(
            np.random.choice([0, 1, 2], 1, p=[1 - (self.env_variables["p_fast"] + self.env_variables["p_slow"]),
                                              self.env_variables["p_slow"],
                                              self.env_variables["p_fast"]])[0])


    def check_proximity_all_prey(self, sensing_distance):
        all_prey_positions = np.array([pr.position for pr in self.prey_bodies])
        fish_position = self.body.position
        within_x = (all_prey_positions[:, 0] > fish_position[0] - sensing_distance) * (all_prey_positions[:, 0] < fish_position[0] + sensing_distance)
        within_y = (all_prey_positions[:, 1] > fish_position[1] - sensing_distance) * (all_prey_positions[:, 1] < fish_position[1] + sensing_distance)
        within_range = within_x * within_y
        return within_range

    def get_last_action_magnitude(self):
        return self.prev_action_impulse * self.env_variables[
            'displacement_scaling_factor']  # Scaled down both for mass effects and to make it possible for the prey to be caught.

    def _move_prey_new(self, micro_step):
        if len(self.prey_bodies) == 0:
            return

        # Generate impulses
        impulse_types = [0, self.env_variables["slow_impulse_paramecia"], self.env_variables["fast_impulse_paramecia"]]
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
                    distance_vector = prey_body.position - self.body.position
                    distance = (distance_vector[0] ** 2 + distance_vector[1] ** 2) ** 0.5
                    distance_scaling = np.exp(-distance)

                    original_angle = copy.copy(prey_body.angle)
                    prey_body.angle = self.body.angle + np.random.uniform(-1, 1)
                    impulse_for_prey = (self.get_last_action_magnitude()/self.env_variables["known_max_fish_i"]) * \
                                        self.env_variables["displacement_scaling_factor"] * distance_scaling

                    prey_body.apply_impulse_at_local_point((impulse_for_prey, 0))
                    prey_body.angle = original_angle

                # Motion from prey escape
                if self.env_variables["prey_jump"] and np.random.choice([0, 1], size=1,
                                                                        p=[1 - self.env_variables["p_escape"],
                                                                           self.env_variables["p_escape"]])[0] == 1:
                    prey_body.apply_impulse_at_local_point((self.env_variables["jump_impulse_paramecia"], 0))

            else:
                if micro_step == 0:
                    prey_body.angle = prey_body.angle + angle_changes[i]

                prey_body.apply_impulse_at_local_point((impulses[i], 0))

    def move_fish(self, impulse, angle):
        self.body.angle += angle
        self.body.apply_impulse_at_local_point((impulse, 0))

    def calculate_impulse(self, distance):
        """
        Uses the derived distance-mass-impulse relationship to convert an input distance (in mm) to impulse
        (arbitrary units).
        :param distance:
        :return:
        """
        # return (distance * 10 - (0.004644 * self.env_variables['fish_mass'] + 0.081417)) / 1.771548
        # return (distance * 10) * 0.360574383  # From mm
        return (distance * 10) * 0.34452532909386484  # From mm

    def check_fish_proximity_to_walls(self):
        fish_position = self.body.position

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

    def create_realistic_predator(self, position):
        self.predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])
        self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_size']/2)
        self.predator_shape.elasticity = 1.0

        fish_position = self.body.position

        x_position = position[0]
        y_position = position[1]

        self.predator_body.position = (x_position, y_position)
        self.predator_target = fish_position
        self.total_predator_steps = 0
        self.predator_body.velocity = (self.predator_target[0]/200, self.predator_target[1]/200)

        self.predator_shape.color = (0, 1, 0)
        self.predator_location = (x_position, y_position)
        self.remaining_predator_attacks = 1 + np.sum(
            np.random.choice([0, 1], self.env_variables["max_predator_attacks"] - 1,
                             p=[1.0 - self.env_variables["further_attack_probability"],
                                self.env_variables["further_attack_probability"]]))

        self.predator_shape.collision_type = 5
        self.predator_shape.filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # Category 2 objects cant collide with predator

        self.space.add(self.predator_body, self.predator_shape)

    def move_realistic_predator(self, micro_step):
        # Update predator target
        self.predator_target = [500, 500] #np.array(self.body.position)

        self.predator_body.angle = np.pi / 2 - np.arctan2(
            self.predator_target[0] - self.predator_body.position[0],
            self.predator_target[1] - self.predator_body.position[1])
        self.predator_body.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def touch_predator(self, arbiter, space, data):
        self.touched_predator = True
        # print(self.micro_step)
        self.body.position = (700, 700)

        return False

    def run_prey_capture(self, prey_position_relative, fixed_capture, continuous, set_impulse, set_angle, num_sim_steps=100):
        # Reset
        self.prey_consumed_this_step = False
        self.prey_bodies = []
        self.prey_shapes = []

        position = []
        self.body.position = (500, 500)
        self.body.angle = 2 * np.pi
        self.body.velocity = (0, 0)

        self.create_prey([500 + prey_position_relative[0], 500 + prey_position_relative[1]])

        # Pre-Movement Resolution (allows effects of prey jumping to be computed...)
        for micro_step in range(num_sim_steps):
            self._move_prey_new(micro_step)
            self.space.step(self.env_variables['phys_dt'])


        # Take fish action
        if continuous:
            self.move_fish(set_impulse, set_angle)
            self.prev_action_impulse = set_impulse
        else:
            if fixed_capture:
                # self.move_fish(2.97, 0)
                self.move_fish(2.1468332, 0)
                self.prev_action_impulse = 2.1468332
            else:
                action_angle, distance = draw_angle_dist_narrowed(0)#, n= 10)  # draw_angle_dist(0)

                action_impulse = self.calculate_impulse(distance)
                # action_angle = np.random.choice([-angle_change, angle_change])
                self.move_fish(action_impulse, action_angle)
                self.prev_action_impulse = action_impulse

        for micro_step in range(num_sim_steps):
            if self.capture_start <= micro_step <= self.capture_end:
                self.capture_possible = True
            else:
                self.capture_possible = False
            position.append(np.array(self.body.position))
            self._move_prey_new(micro_step)
            self.space.step(self.env_variables['phys_dt'])

        position = np.array(position)
        position -= np.array([[500, 500]])
        if self.prey_consumed_this_step:
            return True
        else:
            self.space.remove(self.prey_shapes[-1], self.prey_shapes[-1].body)
            self.prey_bodies.remove(self.prey_shapes[-1].body)
            self.prey_shapes.remove(self.prey_shapes[-1])
            self.paramecia_gaits = []
            return False

        # position = np.array(position)
        # distance = position - np.array([100, 100])
        # distance = (distance[:, 0] ** 2 + distance[:, 1] ** 2) ** 0.5
        # distance = distance/10
        # plt.plot([i for i in range(200)], distance)
        # plt.xlabel("Time (ms)")
        # plt.ylabel("Distance (mm)")
        # plt.vlines(75, ymin=min(distance), ymax=max(distance), color="r")
        # plt.vlines(125, ymin=min(distance), ymax=max(distance), color="r")
        # plt.show()
        # return np.array(self.prey_bodies[0].position)

    def run_predator_escape(self, predator_position_relative, fixed_action, continuous, set_impulse, set_angle, num_sim_steps=100, specified_action=5):
        # Reset
        self.prey_consumed_this_step = False
        self.prey_bodies = []
        self.prey_shapes = []

        self.body.position = (500, 500)
        self.body.angle = 2 * np.pi
        self.body.velocity = (0, 0)

        self.create_realistic_predator([500 + predator_position_relative[0], 500 + predator_position_relative[1]])

        # Take fish action
        if continuous:
            self.move_fish(set_impulse, set_angle)
        else:
            if fixed_action:
                self.move_fish(22.66, 0.33)
            else:
                bout_id = convert_action_to_bout_id(specified_action)
                if specified_action == 6:
                    angle_change, distance = 0, 0
                else:
                    angle_change, distance = draw_angle_dist_narrowed(bout_id)  # draw_angle_dist(0)

                action_impulse = self.calculate_impulse(distance)
                # print(f"Distance: {distance} Impulse; {action_impulse}")
                # action_angle = np.random.choice([-angle_change, angle_change])
                self.move_fish(action_impulse, angle_change)

        pred_position = []
        fish_position = []
        for micro_step in range(num_sim_steps):
            fish_position.append(np.array(self.body.position))
            pred_position.append(np.array(self.predator_body.position))
            # print(self.body.position)
            self.space.step(self.env_variables['phys_dt'])
            self.micro_step = micro_step
            self.move_realistic_predator(micro_step)

        pred_position = np.array(pred_position)
        pred_vectors = pred_position - pred_position[0]
        pred_distance = (pred_vectors[:, 0] ** 2 + pred_vectors[:, 1] ** 2) ** 0.5
        pred_distance = pred_distance[1:] - pred_distance[:-1]
        pred_distance *= 500

        # plt.plot([i*2 for i in range(99)], pred_distance/10)
        # plt.xlabel("Time (ms)")
        # plt.ylabel("Velocity (mms-1)")
        # plt.show()

        pred_final_pos = np.array(self.predator_body.position)
        self.space.remove(self.predator_shape, self.predator_shape.body)
        self.predator_shape = None
        self.predator_body = None

        if self.touched_predator:
            # print(np.array(pred_final_pos))
            self.touched_predator = False
            return False, np.array(fish_position), pred_final_pos
        else:
            return True, np.array(fish_position), pred_final_pos

    def run_prey_motion(self, num_steps=10, num_sim_steps=100):
        # Reset
        self.prey_consumed_this_step = False
        self.prey_bodies = []
        self.prey_shapes = []

        self.body.position = (1000, 1000)
        self.body.angle = 2 * np.pi
        self.body.velocity = (0, 0)

        self.create_prey([500, 500])

        prey_position = []

        for step in range(num_steps):
            for micro_step in range(num_sim_steps):
                self._move_prey_new(micro_step)
                prey_position.append(np.array(self.prey_bodies[-1].position))
                self.space.step(self.env_variables['phys_dt'])

        prey_position = np.array(prey_position)
        prey_position -= np.array([[500, 500]])
        prey_position /= 10

        prey_distance = (prey_position[:, 0] ** 2 + prey_position[:, 1] ** 2) ** 0.5

        prey_velocity = prey_distance[1:] - prey_distance[:-1]
        prey_velocity *= 5
        prey_velocity *= 100

        fig, axs = plt.subplots(3)
        axs[0].plot(prey_position[:, 0], prey_position[:, 1])
        axs[0].set_ylabel("Prey Position (mm from starting point")
        axs[0].set_xlabel("Prey Position (mm from starting point")
        axs[1].plot([i *(2/1000) for i in range(prey_distance.shape[0])], prey_distance)
        axs[1].set_ylabel("Prey Distance (mm)")
        axs[1].set_xlabel("Time (s)")
        axs[2].plot([i*(2/1000) for i in range(prey_velocity.shape[0])], prey_velocity)
        axs[2].set_ylabel("Prey Velocity (mms-1)")
        axs[2].set_xlabel("Time (s)")
        plt.show()

        self.space.remove(self.prey_shapes[-1], self.prey_shapes[-1].body)
        self.prey_bodies.remove(self.prey_shapes[-1].body)
        self.prey_shapes.remove(self.prey_shapes[-1])

