import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import random

import pymunk
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from Environment.Action_Space.draw_angle_dist_new import draw_angle_dist_new as draw_angle_dist_narrowed
from Environment.Action_Space.plot_bout_data import get_bout_data
from Environment.Action_Space.draw_angle_dist import convert_action_to_bout_id
from Analysis.Behavioural.VisTools.get_action_name import get_action_name_unlateralised


class TestEnvironment:

    def __init__(self, predator_impulse):
        self.touched_predator = False

        self.env_variables = {
            'phys_dt': 0.2,  # physics time step
            'drag': 0.7,  # water drag

            'fish_mass': 140.,
            'fish_mouth_size': 4.,  # FINAL VALUE - 0.2mm diameter, so 1.
            'fish_head_size': 2.5,  # Old - 10
            'fish_tail_length': 41.5,  # Old: 70
            'eyes_verg_angle': 77.,  # in deg

            'prey_mass': 140.,
            'prey_inertia': 40.,
            'prey_size': 2.5,  # FINAL VALUE - 0.2mm diameter, so 1.
            'prey_max_turning_angle': 0.04,

            'p_slow': 1.0,
            'p_fast': 0.0,
            'p_escape': 0.0,
            'p_switch': 0.0,  # Corresponds to 1/average duration of movement type.
            'p_reorient': 0.001,
            'slow_speed_paramecia': 0.0037,  # Impulse to generate 0.5mms-1 for given prey mass
            'fast_speed_paramecia': 0.0074,  # Impulse to generate 1.0mms-1 for given prey mass
            'jump_speed_paramecia': 0.074,  # Impulse to generate 10.0mms-1 for given prey mass
            'prey_fluid_displacement': True,

            'predator_mass': 200.,
            'predator_inertia': 0.0001,
            'predator_size': 32, #87.,  # To be 8.7mm in diameter, formerly 100
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
        inertia = pymunk.moment_for_circle(self.env_variables['fish_mass'], 0, self.env_variables['fish_head_size'],
                                           (0, 0))
        self.body = pymunk.Body(1, inertia)
        # Mouth
        self.mouth = pymunk.Circle(self.body, self.env_variables['fish_mouth_size'], offset=(0, 0))
        self.mouth.color = (0, 1, 0)
        self.mouth.elasticity = 1.0
        self.mouth.collision_type = 3

        # Head
        self.head = pymunk.Circle(self.body, self.env_variables['fish_head_size'],
                                  offset=(-self.env_variables['fish_head_size'], 0))
        self.head.color = (0, 1, 0)
        self.head.elasticity = 1.0
        self.head.collision_type = 6

        # # Tail
        tail_coordinates = ((-self.env_variables['fish_head_size'], 0),
                            (-self.env_variables['fish_head_size'], - self.env_variables['fish_head_size']),
                            (-self.env_variables['fish_head_size'] - self.env_variables['fish_tail_length'], 0),
                            (-self.env_variables['fish_head_size'], self.env_variables['fish_head_size']))
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


        self.prey_consumed_this_step = False

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
        # print(f"Touched: {self.body.position} - {self.prey_bodies[0].position}")
        # self.prey_consumed_this_step = True
        # return True
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
        self.paramecia_gaits.append(1)

    def _move_prey_new(self, micro_step):
        # gaits_to_switch = np.random.choice([0, 1], len(self.prey_shapes),
        #                                    p=[1 - self.env_variables["p_switch"], self.env_variables["p_switch"]])
        # switch_to = np.random.choice([0, 1, 2], len(self.prey_shapes),
        #                              p=[1 - (self.env_variables["p_slow"] + self.env_variables["p_fast"]),
        #                                 self.env_variables["p_slow"], self.env_variables["p_fast"]])
        # self.paramecia_gaits = [switch_to[i] if gaits_to_switch[i] else old_gait for i, old_gait in
        #                         enumerate(self.paramecia_gaits)]

        # Generate impulses
        impulse_types = [0, self.env_variables["slow_speed_paramecia"], self.env_variables["fast_speed_paramecia"]]
        impulses = [impulse_types[gait] for gait in self.paramecia_gaits]

        # Angles of change - can generate as same for all.
        if micro_step == 0:
            angle_changes = np.random.uniform(-self.env_variables['prey_max_turning_angle'],
                                              self.
                                              env_variables['prey_max_turning_angle'],
                                              len(self.prey_shapes))

        for i, prey_body in enumerate(self.prey_bodies):
            # if self.check_proximity(prey_body.position, self.env_variables['prey_sensing_distance']):
            #     # Motion from fluid dynamics
            #     if self.env_variables["prey_fluid_displacement"]:
            #         original_angle = copy.copy(prey_body.angle)
            #         prey_body.angle = self.fish.body.angle + np.random.uniform(-1, 1)
            #         prey_body.apply_impulse_at_local_point((self.get_last_action_magnitude(), 0))
            #         prey_body.angle = original_angle
            #
            #     # Motion from prey escape
            #     if self.env_variables["prey_jump"] and np.random.choice(1, [0, 1],
            #                                                             p=[1 - self.env_variables["p_escape"],
            #                                                                self.env_variables["p_escape"]])[0] == 1:
            #         prey_body.apply_impulse_at_local_point((self.env_variables["jump_speed_paramecia"], 0))
            #
            # else:
            #     prey_body.angle = prey_body.angle + angle_changes[i]
            prey_body.apply_impulse_at_local_point((impulses[i], 0))
            if micro_step == 0:
                prey_body.angle = prey_body.angle + angle_changes[i]

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

    def run(self, predator_position_relative, fixed_action, continuous, set_impulse, set_angle, num_sim_steps=100, specified_action=5):
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
                distance, angle_change = draw_angle_dist_narrowed(bout_id)  # draw_angle_dist(0)

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

        plt.plot([i*2 for i in range(99)], pred_distance/10)
        plt.xlabel("Time (ms)")
        plt.ylabel("Velocity (mms-1)")
        plt.show()

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


def plot_escape_success(n_repeats=1, use_action_means=True,
                        continuous=False, set_impulse=2., set_angle=0., set_action=True, impulse_effect_noise=0.,
                        angular_effect_noise=0., predator_impulse=2., specified_action=7):
    env = TestEnvironment(predator_impulse)
    xp, yp = np.arange(-100, 100, 10), np.arange(-100, 100, 10)
    resolution = 1

    xpe, ype = np.meshgrid(xp, yp)
    vectors1 = np.concatenate((np.expand_dims(xpe, 2), np.expand_dims(ype, 2)), axis=2)
    vectors = np.reshape(vectors1, (-1, 2))
    successful_escape_count = np.zeros((vectors.shape[0]))
    pred_final_positions = []

    for j in range(n_repeats):
        print(f"{j} / {n_repeats}")
        for i, vector in enumerate(vectors):
            # print(f"{i} / {n_test}")
            # Apply motor effect noise
            if continuous:
                if impulse_effect_noise > 0:
                    impulse = set_impulse + (np.random.normal(0, impulse_effect_noise) * abs(set_impulse))
                else:
                    impulse = set_impulse
                if angular_effect_noise > 0:
                    angle = set_angle + (np.random.normal(0, angular_effect_noise) * abs(set_angle))
                else:
                    angle = set_angle
            else:
                if set_action:
                    impulse = set_impulse
                    angle = set_angle
                else:
                    impulse = None
                    angle = None

            s, fish_pos, pred_pos = env.run(vector, fixed_action=use_action_means, continuous=continuous, set_impulse=impulse,
                        set_angle=angle, specified_action=specified_action)
            pred_final_positions.append(pred_pos)

            if s:
                successful_escape_count[i] += 1

    successful_escape_count = np.reshape(successful_escape_count, (vectors1.shape[0], vectors1.shape[1]))
    successful_escape_count /= n_repeats

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(successful_escape_count)

    # Display fish.
    x, y = 10, 10
    mouth_size, head_size, tail_length = env.env_variables['fish_mouth_size'], env.env_variables['fish_head_size'], \
                                         env.env_variables['fish_tail_length']
    mouth_size *= resolution/10
    head_size *= resolution/10
    tail_length *= resolution/10

    mouth_centre = (x, y)
    mouth = plt.Circle(mouth_centre, mouth_size, fc="green")
    ax.add_patch(mouth)

    angle = (1.5 * np.pi)
    dx1, dy1 = head_size * np.sin(angle), head_size * np.cos(angle)
    head_centre = (mouth_centre[0] + dx1,
                   mouth_centre[1] + dy1)
    head = plt.Circle(head_centre, head_size, fc="green")
    ax.add_patch(head)

    dx2, dy2 = -1 * dy1, dx1
    left_flank = (head_centre[0] + dx2,
                  head_centre[1] + dy2)
    right_flank = (head_centre[0] - dx2,
                   head_centre[1] - dy2)
    tip = (mouth_centre[0] + (tail_length + head_size) * np.sin(angle),
           mouth_centre[1] + (tail_length + head_size) * np.cos(angle))
    tail = plt.Polygon(np.array([left_flank, right_flank, tip]), fc="green")
    ax.add_patch(tail)

    # Predator
    predator_size = 32 * resolution/10
    dx1, dy1 = head_size * np.sin(angle), head_size * np.cos(angle)
    predator_centre = (5 + dx1,
                       5 + dy1)
    predator = plt.Circle(predator_centre, predator_size/2, fc="red")
    ax.add_patch(predator)

    scale_bar = AnchoredSizeBar(ax.transData,
                                resolution, '1mm', 'lower right',
                                pad=1,
                                color='red',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 16}
                                )
    ax.add_artist(scale_bar)
    fig.colorbar(im, ax=ax, label="Prop Success")
    fish_pos /= 10/resolution
    fish_pos -= 50
    fish_pos += 10
    #
    # pred_final_positions /= 10/resolution
    # pred_final_positions -= 50
    # pred_final_positions += 40

    plt.scatter(fish_pos[:, 0], fish_pos[:, 1])
    plt.title(f"Bout: {get_action_name_unlateralised(specified_action)} ")
    # plt.scatter(pred_final_positions[:, 0], pred_final_positions[:, 1], alpha=0.1)
    plt.show()


if __name__ == "__main__":
    # plot_escape_success(n_repeats=1,
    #                     use_action_means=False, continuous=True, set_impulse=0,
    #                     set_angle=0.,
    #                     impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=25)
    # plot_escape_success(n_repeats=1,
    #                     use_action_means=True, continuous=False, set_impulse=0,
    #                     set_angle=0.0,
    #                     impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=15., specified_action=6)
    # plot_escape_success(n_repeats=1,
    #                     use_action_means=False, continuous=False, set_impulse=0,
    #                     set_angle=0.0,
    #                     impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=25., specified_action=1)
    plot_escape_success(n_repeats=1,
                        use_action_means=False, continuous=False, set_impulse=0,
                        set_angle=0.0,
                        impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=25., specified_action=7)
    # plot_escape_success(n_repeats=20,
    #                     use_action_means=False, continuous=False, set_impulse=0,
    #                     set_angle=0.0,
    #                     impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=25., specified_action=1)
    # plot_escape_success(n_repeats=1,
    #                     use_action_means=False, continuous=True, set_impulse=0,
    #                     set_angle=0.,
    #                     impulse_effect_noise=0.0, angular_effect_noise=0.0, predator_impulse=35)
