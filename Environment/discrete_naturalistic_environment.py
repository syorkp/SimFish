import numpy as np
import matplotlib.pyplot as plt

from Environment.naturalistic_environment import NaturalisticEnvironment
from Environment.Fish.fish import Fish


class DiscreteNaturalisticEnvironment(NaturalisticEnvironment):

    def __init__(self, env_variables, realistic_bouts, using_gpu, draw_screen=False, fish_mass=None,
                 collisions=True, relocate_fish=None, num_actions=10, run_version=None, split_event=None,
                 modification=None):

        super().__init__(env_variables, realistic_bouts, using_gpu, draw_screen, fish_mass, collisions,
                         relocate_fish, num_actions=num_actions, run_version=run_version, split_event=split_event,
                         modification=modification)

        # Create the fish class instance and add to the space.
        if fish_mass is None:
            self.fish = Fish(self.board, env_variables, self.dark_col, realistic_bouts, using_gpu)
        else:
            # In the event that I am producing a calibration curve for distance moved.
            self.fish = Fish(self.board, env_variables, self.dark_col, realistic_bouts, using_gpu,
                             fish_mass=fish_mass)

        self.space.add(self.fish.body, self.fish.mouth, self.fish.head, self.fish.tail)

        # Create walls.
        self.create_walls()
        self.reset()

        self.col = self.space.add_collision_handler(2, 3)
        self.col.begin = self.touch_prey

        if collisions:
            self.pred_col = self.space.add_collision_handler(5, 3)
            self.pred_col.begin = self.touch_predator
            self.pred_col2 = self.space.add_collision_handler(5, 6)
            self.pred_col2.begin = self.touch_predator

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_wall

        self.edge_pred_col = self.space.add_collision_handler(1, 5)
        self.edge_pred_col.begin = self.remove_realistic_predator

        self.grain_fish_col = self.space.add_collision_handler(3, 4)
        self.grain_fish_col.begin = self.touch_grain

        # to prevent predators from knocking out prey  or static grains
        self.grain_pred_col = self.space.add_collision_handler(4, 5)
        self.grain_pred_col.begin = self.no_collision
        self.prey_pred_col = self.space.add_collision_handler(2, 5)
        self.prey_pred_col.begin = self.no_collision

        # To prevent the differential wall being hit by fish
        self.fish_prey_wall = self.space.add_collision_handler(3, 7)
        self.fish_prey_wall.begin = self.no_collision
        self.fish_prey_wall2 = self.space.add_collision_handler(6, 7)
        self.fish_prey_wall2.begin = self.no_collision
        self.pred_prey_wall2 = self.space.add_collision_handler(5, 7)
        self.pred_prey_wall2.begin = self.no_collision

        self.cs_required = self.env_variables["cs_required"]

        self.continuous_actions = False

    def reset(self):
        self.cs_required = self.env_variables["cs_required"]
        super().reset()

    def simulation_step(self, action, activations=None, impulse=None):
        if self.cs_required:
            self.fish.making_capture = False
        else:
            self.fish.making_capture = True
        return super().simulation_step(action, activations, impulse)

    def load_simulation(self, buffer, background, energy_state):
        self.fish.prev_action_impulse = buffer.efference_copy_buffer[-1][0][1]
        self.fish.prev_action_angle = buffer.internal_state_buffer[-1][0][2]
        super().load_simulation(buffer, background, energy_state)
