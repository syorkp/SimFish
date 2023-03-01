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
            'phys_dt': 0.2,  # physics time step
            'drag': 0.7,  # water drag

            'fish_mass': 140.,
            'fish_mouth_radius': 4.,  # FINAL VALUE - 0.2mm diameter, so 1.
            'fish_head_radius': 2.5,  # Old - 10
            'fish_tail_length': 41.5,  # Old: 70
            'eyes_verg_angle': 77.,  # in deg

            'prey_mass': 140.,
            'prey_inertia': 40.,
            'prey_radius': 2.5,  # FINAL VALUE - 0.2mm diameter, so 1.
            'prey_max_turning_angle': 0.04,

            'p_slow': 1.0,
            'p_fast': 0.0,
            'p_escape': 0.0,
            'p_switch': 0.0,  # Corresponds to 1/average duration of movement type.
            'p_reorient': 0.001,
            'slow_impulse_paramecia': 0.0037,  # Impulse to generate 0.5mms-1 for given prey mass
            'fast_impulse_paramecia': 0.0074,  # Impulse to generate 1.0mms-1 for given prey mass
            'jump_impulse_paramecia': 0.074,  # Impulse to generate 10.0mms-1 for given prey mass
            'prey_fluid_displacement': True,

            'predator_mass': 10.,
            'predator_inertia': 40.,
            'predator_radius': 87.,  # To be 8.7mm in diameter, formerly 100
            'predator_impulse': 1.0,
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

        self.fraction_capture_possible, self.permitted_angular_deviation = fraction_capture_possible, permitted_angular_deviation

        self.capture_fraction = int(
            100 * fraction_capture_possible)
        self.capture_start = 0 #int((100 - self.capture_fraction) / 2)
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

                        deviation = abs((2 * np.pi) - (fish_orientation + angle))
                    if deviation < self.permitted_angular_deviation:
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
        old = (distance * 10 - (0.004644 * self.env_variables['fish_mass'] + 0.081417)) / 1.771548
        new = (distance * 10) * 0.360574383  # From mm
        newest = (distance * 10) * 0.34452532909386484  # From mm

        print(f"OLD: {old}  NEW: {new}   NEWEST: {newest}")

    def run(self, impulse, num_sim_steps=100):
        self.body.position = (500, 500)
        self.body.angle = 2 * np.pi
        self.body.velocity = (0, 0)

        # Take fish action
        angle = 0
        self.move_fish(impulse, angle)
        
        for micro_step in range(num_sim_steps):
            self.space.step(self.env_variables['phys_dt'])

        position_after = self.body.position

        motion_vector = np.array(position_after) - np.array([500, 500])

        distance = (motion_vector[0] ** 2 + motion_vector[1] ** 2) ** 0.5
        return distance


if __name__ == "__main__":
    env = TestEnvironment(0, 0)
    impulses_to_test = np.linspace(0, 100, 1000)
    distances = [env.run(i, num_sim_steps=100) / 10 for i in impulses_to_test]
    plt.plot(impulses_to_test, distances)
    plt.show()
    p = np.polyfit(distances, impulses_to_test, 1)
    for d in distances:
        env.calculate_impulse(d)
        