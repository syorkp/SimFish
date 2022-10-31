import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

import pymunk
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from Environment.Action_Space.draw_angle_dist import draw_angle_dist, draw_angle_dist_narrowed
from Environment.Action_Space.plot_bout_data import get_bout_data


class TestEnvironment:

    def __init__(self, fraction_capture_possible, permitted_angular_deviation):
        self.env_variables = {
            'phys_dt': 0.1,  # physics time step
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

            'predator_mass': 10.,
            'predator_inertia': 40.,
            'predator_size': 87.,  # To be 8.7mm in diameter, formerly 100
            'predator_impulse': 1.0,
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

        self.fraction_capture_possible, self.permitted_angular_deviation = fraction_capture_possible, permitted_angular_deviation

        self.capture_fraction = int(
            100 * fraction_capture_possible)
        self.capture_start = int((100 - self.capture_fraction) / 2)
        self.capture_end = self.capture_start + self.capture_fraction

        self.prey_consumed_this_step = False

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
        return (distance * 10) * 0.360574383  # From mm


    def run(self, prey_position_relative, fixed_capture, continuous, set_impulse, set_angle, num_sim_steps=100):
        # Reset
        self.prey_consumed_this_step = False
        self.prey_bodies = []
        self.prey_shapes = []

        position = []
        self.body.position = (500, 500)
        self.body.angle = 2 * np.pi
        self.body.velocity = (0, 0)

        self.create_prey([500 + prey_position_relative[0], 500 + prey_position_relative[1]])

        # Take fish action
        if continuous:
            self.move_fish(set_impulse, set_angle)
        else:
            if fixed_capture:
                # self.move_fish(2.97, 0)
                self.move_fish(2.1468332, 0)
            else:
                angle_change, distance = draw_angle_dist_narrowed(0, n= 10)  # draw_angle_dist(0)

                action_impulse = self.calculate_impulse(distance)
                action_angle = np.random.choice([-angle_change, angle_change])
                self.move_fish(action_impulse, action_angle)

        for micro_step in range(num_sim_steps):
            if self.capture_start <= micro_step <= self.capture_end:
                self.capture_possible = True
            else:
                self.capture_possible = False
            position.append(np.array(self.body.position))
            self.space.step(self.env_variables['phys_dt']*2)

        position = np.array(position)
        position -= np.array([[500, 500]])
        if self.prey_consumed_this_step:
            return True
        else:
            self.space.remove(self.prey_shapes[-1], self.prey_shapes[-1].body)
            self.prey_bodies.remove(self.prey_shapes[-1].body)
            self.prey_shapes.remove(self.prey_shapes[-1])
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


def plot_strike_zone(fraction_capture_permitted, angle_deviation_allowed, n_repeats=1, use_action_means=True,
                     continuous=False, overlay_all_sCS_data=False, set_impulse=2., set_angle=0., impulse_effect_noise=0.,
                     angular_effect_noise=0.):
    env = TestEnvironment(fraction_capture_permitted, angle_deviation_allowed)

    xp, yp = np.arange(-5, 30, 0.5), np.arange(-8, 8, 0.5)
    resolution = 20

    xpe, ype = np.meshgrid(xp, yp)
    vectors1 = np.concatenate((np.expand_dims(xpe, 2), np.expand_dims(ype, 2)), axis=2)
    vectors = np.reshape(vectors1, (-1, 2))
    successful_capture_count = np.zeros((vectors.shape[0]))
    angs = np.zeros((vectors.shape[0]))
    n_test = vectors.shape[0]

    fish_positions = []

    for j in range(n_repeats):
        print(f"{j} / {n_repeats}")
        for i, vector in enumerate(vectors):
            # print(f"{i} / {n_test}")
            # Apply motor effect noise
            if impulse_effect_noise > 0:
                impulse = set_impulse + (np.random.normal(0, impulse_effect_noise) * abs(set_impulse))
            else:
                impulse = set_impulse
            if angular_effect_noise > 0:
                angle = set_angle + (np.random.normal(0, angular_effect_noise) * abs(set_angle))
            else:
                angle = set_angle

            s = env.run(vector, fixed_capture=use_action_means, continuous=continuous, set_impulse=impulse,
                        set_angle=angle)

            if s:
                # fish_positions.append(np.array(env.body.position))
                successful_capture_count[i] += 1
                angs[i] = env.latest_incidence

    for j in range(n_repeats):
        print(f"{j} / {n_repeats}")
        for i, vector in enumerate(vectors):
            # print(f"{i} / {n_test}")
            # Apply motor effect noise
            if impulse_effect_noise > 0:
                impulse = set_impulse + (np.random.normal(0, impulse_effect_noise) * abs(set_impulse))
            else:
                impulse = set_impulse
            if angular_effect_noise > 0:
                angle = set_angle + (np.random.normal(0, angular_effect_noise) * abs(set_angle))
            else:
                angle = set_angle

            s = env.run([0, 0], fixed_capture=use_action_means, continuous=continuous, set_impulse=impulse,
                        set_angle=angle)

            fish_positions.append(np.array(env.body.position))

    fish_positions = np.array(fish_positions)
    successful_capture_count = np.reshape(successful_capture_count, (vectors1.shape[0], vectors1.shape[1]))
    angs = np.reshape(angs, (vectors1.shape[0], vectors1.shape[1]))
    successful_capture_count /= n_repeats

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(successful_capture_count)

    # Display fish.
    x, y = 10, 16
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

    scale_bar = AnchoredSizeBar(ax.transData,
                                resolution, '1mm', 'lower right',
                                pad=1,
                                color='red',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 16}
                                )
    ax.add_artist(scale_bar)

    fish_positions -= 500
    fish_positions[:, 0] = np.absolute(fish_positions[:, 0])
    fish_positions *= resolution/10
    fish_positions[:, 0] += x
    fish_positions[:, 1] += y

    # plt.scatter(fish_positions[:, 0], fish_positions[:, 1], alpha=0.3)

    if overlay_all_sCS_data:
        distances, angles = get_bout_data(3)
        x_diff = distances * np.sin(angles)
        y_diff = distances * np.cos(angles)

        x_loc = x + (y_diff * resolution)
        y_loc = y + (x_diff * resolution)

        # Mirror yloc
        y_loc2 = y - (x_diff * resolution)
        x_loc = np.concatenate((x_loc, x_loc))
        y_loc = np.concatenate((y_loc, y_loc2))
        density_list = np.concatenate((np.expand_dims(x_loc, 1), np.expand_dims(y_loc, 1)), axis=1)

        plt.scatter(x_loc, y_loc, alpha=0.3)

        # Show density
        x = np.array([i[0] for i in density_list])
        y = np.array([i[1] for i in density_list])
        y = np.negative(y)
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        nbins = 300
        k = kde.gaussian_kde([y, x])
        yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="Reds", )  # cmap='gist_gray')#  cmap='PuBu_r')
        # plt.contour(yi, -xi, zi.reshape(xi.shape), 3)
        #
        # plt.xlim(0, successful_capture_count.shape[1]-1)
        # plt.ylim(0, successful_capture_count.shape[0]-1)


    fig.colorbar(im, ax=ax, label="Prop Success")

    plt.show()

    return angs
    # Run for many different drawn parameters, plotting scatter of whether was successful capture.
    # Run the same but in the reverse - not var


if __name__ == "__main__":
    # create_video_col(fraction_capture_permitted=0.8, angle_deviation_allowed=np.pi / 8,
    #                  use_action_means=False, continuous=True, set_angle=0.5)
    # plot_strike_zone(fraction_capture_permitted=0.5, angle_deviation_allowed=0.5934119456780723, n_repeats=1,
    #                  use_action_means=False, continuous=False)
    # plot_strike_zone(fraction_capture_permitted=1, angle_deviation_allowed=np.pi / 8, n_repeats=100,
    #                  use_action_means=False, continuous=False, overlay_all_sCS_data=True)
    # angs = plot_strike_zone(fraction_capture_permitted=0.8, angle_deviation_allowed=np.pi / 8, n_repeats=10,
    #                         use_action_means=False, continuous=True, overlay_all_sCS_data=True,# set_impulse=2.1,
    #                         set_angle=0.4,
    #                         impulse_effect_noise=0.98512558, angular_effect_noise=0.0010472)
    # angs = plot_strike_zone(fraction_capture_permitted=0.8, angle_deviation_allowed=np.pi / 8, n_repeats=10,
    #                         use_action_means=False, continuous=True, overlay_all_sCS_data=True,# set_impulse=2.1,
    #                         set_angle=0.4,
    #                         impulse_effect_noise=0.14, angular_effect_noise=0.5)
    plot_strike_zone(fraction_capture_permitted=0.98, angle_deviation_allowed=np.pi,#,/8,
                     n_repeats=1,
                     use_action_means=True, continuous=False, overlay_all_sCS_data=True)

