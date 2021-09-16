import numpy as np
import matplotlib.pyplot as plt

from Environment.naturalistic_environment import NaturalisticEnvironment
from Environment.Fish.fish import Fish
from Environment.Fish.continuous_fish import ContinuousFish


class ContinuousNaturalisticEnvironment(NaturalisticEnvironment):

    def __init__(self, env_variables, realistic_bouts, draw_screen=False, fish_mass=None, collisions=True):
        # Create the fish class instance and add to the space.
        if fish_mass is None:
            self.fish = ContinuousFish(self.board, env_variables, self.dark_col, realistic_bouts)
        else:
            # In the event that I am producing a calibration curve for distance moved.
            self.fish = ContinuousFish(self.board, env_variables, self.dark_col, realistic_bouts, fish_mass=fish_mass)
        super().__init__(env_variables, realistic_bouts, draw_screen, fish_mass, collisions)

    def reset(self):
        super().reset()

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None, impulse=None):
        self.fish.making_capture = True
        return super().simulation_step(action, save_frames, frame_buffer, activations, impulse)

