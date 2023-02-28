import numpy as np
from Environment.Fish.fish import Fish


class ContinuousFish(Fish):

    def __init__(self, board, env_variables, dark_col, using_gpu, fish_mass=None):
        super().__init__(board=board,
                         env_variables=env_variables,
                         dark_col=dark_col,
                         using_gpu=using_gpu,
                         fish_mass=fish_mass
                         )

        self.making_capture = True

    @staticmethod
    def calculate_distance(impulse):
        return impulse/(10 * 0.34452532909386484)  # To mm

    def take_action(self, action):
        impulse = action[0]
        angle = action[1]

        impulse_deviation = (np.random.normal(0, self.env_variables["impulse_effect_noise_sd_x"]) * abs(impulse)) + \
                            (np.random.normal(0, self.env_variables["impulse_effect_noise_sd_c"]) * abs(angle))

        angle_deviation = (np.random.normal(0, self.env_variables["angle_effect_noise_sd_x"]) * abs(angle)) + \
                          (np.random.normal(0, self.env_variables["angle_effect_noise_sd_c"]) * abs(impulse))

        impulse = impulse + impulse_deviation
        angle = angle + angle_deviation

        self.prev_action_impulse = impulse
        self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))

        self.body.angle += angle
        return 0.0
