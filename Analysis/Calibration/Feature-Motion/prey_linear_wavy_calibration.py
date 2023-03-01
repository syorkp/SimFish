"""
Finding relationship between prey impulse and distance moved.
"""
import copy

import numpy as np
import pymunk
import matplotlib.pyplot as plt


class TestEnvironment:

    def __init__(self):
        self.env_variables = {
            'prey_mass': 1.,
            'phys_dt': 0.1,  # physics time step
            'drag': 0.7,  # water drag

            'prey_inertia': 40.,
            'prey_radius': 1.,  # FINAL VALUE - 0.2mm diameter, so 1.
            'prey_max_turning_angle': 0.3,

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
            "displacement_scaling_factor": 0.018,
        }
        self.prey_bodies = []
        self.prey_shapes = []
        self.paramecia_gaits = []

        self.predator_bodies = []
        self.predator_shapes = []

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = self.env_variables['drag']

    def create_prey(self, prey_position=None):
        self.prey_bodies.append(pymunk.Body(self.env_variables['prey_mass'], self.env_variables['prey_inertia']))
        self.prey_shapes.append(pymunk.Circle(self.prey_bodies[-1], self.env_variables['prey_radius']))
        self.prey_shapes[-1].elasticity = 1.0
        self.prey_bodies[-1].position = prey_position
        self.prey_shapes[-1].color = (0, 0, 1)
        self.prey_shapes[-1].collision_type = 2
        self.prey_bodies[-1].angle = np.pi/4
        # self.prey_shapes[-1].filter = pymunk.ShapeFilter(
        #     mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.prey_bodies[-1], self.prey_shapes[-1])

        # New prey motion TODO: Check doesnt mess with base version.
        # self.paramecia_gaits.append(
        #     np.random.choice([0, 1, 2], 1, p=[1 - (self.env_variables["p_fast"] + self.env_variables["p_slow"]),
        #                                       self.env_variables["p_slow"],
        #                                       self.env_variables["p_fast"]])[0])
        self.paramecia_gaits.append(1)

    def _move_prey_new(self, micro_step, fish_impulse):
            # gaits_to_switch = np.random.choice([0, 1], len(self.prey_shapes),
            #                                    p=[1 - self.env_variables["p_switch"], self.env_variables["p_switch"]])
            # switch_to = np.random.choice([0, 1, 2], len(self.prey_shapes),
            #                              p=[1 - (self.env_variables["p_slow"] + self.env_variables["p_fast"]),
            #                                 self.env_variables["p_slow"], self.env_variables["p_fast"]])
            # self.paramecia_gaits = [switch_to[i] if gaits_to_switch[i] else old_gait for i, old_gait in
            #                         enumerate(self.paramecia_gaits)]

            # Generate impulses
            impulse_types = [0, self.env_variables["slow_impulse_paramecia"], self.env_variables["fast_impulse_paramecia"]]
            impulses = [impulse_types[gait] for gait in self.paramecia_gaits]

            # Angles of change - can generate as same for all.
            if micro_step == 0:
                angle_changes = np.random.uniform(-self.env_variables['prey_max_turning_angle'],
                                                  self.
                                                  env_variables['prey_max_turning_angle'],
                                                  len(self.prey_shapes))

            for i, prey_body in enumerate(self.prey_bodies):
                if fish_impulse:
                    distance_scaling = np.exp(-1)
                    original_angle = copy.copy(prey_body.angle)
                    prey_body.angle = original_angle + 3/2
                    impulse_for_prey = (5/20) * \
                                        self.env_variables["displacement_scaling_factor"] * distance_scaling
                    prey_body.apply_impulse_at_local_point((impulse_for_prey, 0))
                    prey_body.angle = original_angle

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
                #         prey_body.apply_impulse_at_local_point((self.env_variables["jump_impulse_paramecia"], 0))
                #
                # else:
                #     prey_body.angle = prey_body.angle + angle_changes[i]
                prey_body.apply_impulse_at_local_point((impulses[i], 0))
                if micro_step == 0:
                    prey_body.angle = prey_body.angle + angle_changes[i]

    def create_predator(self, prey_position=None):
        self.predator_bodies.append(pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia']))
        self.predator_shapes.append(pymunk.Circle(self.predator_bodies[-1], self.env_variables['predator_radius']))
        self.predator_shapes[-1].elasticity = 1.0
        self.predator_bodies[-1].position = prey_position
        self.predator_shapes[-1].color = (0, 0, 1)
        self.predator_shapes[-1].collision_type = 2
        # self.prey_shapes[-1].filter = pymunk.ShapeFilter(
        #     mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.predator_bodies[-1], self.predator_shapes[-1])

    def move_predator(self):
        self.predator_bodies[0].apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def run(self, num_sim_steps=200, fish_impulse=False):
        self.create_prey([100, 100])
        # self.create_predator([100, 100])
        position = []
        for micro_step in range(num_sim_steps):
            self._move_prey_new(micro_step, fish_impulse)
            # fish_impulse = False
            # self.move_predator()
            position.append(np.array(self.prey_bodies[0].position))
            self.space.step(self.env_variables['phys_dt'])
        position = np.array(position)
        distance = position - np.array([100, 100])
        distance = (distance[:, 0] ** 2 + distance[:, 1] ** 2) ** 0.5
        # plt.plot(distance)
        # plt.show()
        return position



np.random.seed(303)
env = TestEnvironment()
positions = np.zeros((0, 2))
fish_impulse = False
for i in range(50):
    pos = env.run(200, fish_impulse)
    positions = np.concatenate((np.array(positions), pos), axis=0)


positions = np.array(positions)
positions /= 10
plt.scatter(positions[:, 0], positions[:, 1])
plt.axis("scaled")
plt.xlabel("X Distance (mm)")
plt.ylabel("Y Distance (mm)")
plt.show()

np.random.seed(303)
env = TestEnvironment()
positions = np.zeros((0, 2))
fish_impulse = False
for i in range(50):
    if i == 25:
        fish_impulse = True
    pos = env.run(200, fish_impulse)
    positions = np.concatenate((np.array(positions), pos), axis=0)
    fish_impulse = False


positions = np.array(positions)
positions /= 10
plt.scatter(positions[:, 0], positions[:, 1])
plt.axis("scaled")
plt.xlabel("X Distance (mm)")
plt.ylabel("Y Distance (mm)")
plt.show()