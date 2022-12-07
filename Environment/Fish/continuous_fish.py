import numpy as np
from Environment.Fish.fish import Fish


class ContinuousFish(Fish):

    def __init__(self, board, env_variables, dark_col, realistic_bouts, new_simulation, using_gpu, fish_mass=None):
        super().__init__(board, env_variables, dark_col, realistic_bouts, new_simulation, using_gpu,
                         fish_mass=fish_mass)

        self.making_capture = True
        self.new_simulation = new_simulation

    def calculate_distance(self, impulse):
        # return (1.771548 * impulse + self.env_variables['fish_mass'] * 0.004644 + 0.081417) / 10
        return impulse/(10 * 0.34452532909386484)  # To mm

    def take_action(self, action):
        """Overrides method for discrete fish"""
        if self.new_simulation:
            return self._take_action_new(action)
        else:
            return self._take_action(action)

    def _take_action(self, action):
        impulse = action[0]
        angle = action[1]
        self.prev_action_impulse = impulse
        self.prev_action_angle = angle

        distance = self.calculate_distance(impulse)
        reward = - self.calculate_action_cost(angle, distance) - self.env_variables['baseline_penalty']
        self.prev_action_impulse = impulse
        self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
        self.body.angle += angle
        return reward

    def _take_action_new(self, action):
        impulse = action[0]
        angle = action[1]

        # ORIGINAL
        # Noise from uniform.
        # impulse_deviation = np.absolute(np.random.normal(0, self.env_variables["impulse_effect_noise_sd_x"])) * abs(impulse) + \
        #                     self.env_variables["impulse_effect_noise_sd_c"]
        # impulse_deviation = impulse_deviation.item()
        # impulse = impulse + impulse_deviation
        #
        # angle_deviation = np.absolute(np.random.normal(0, self.env_variables["angle_effect_noise_sd_x"])) * abs(angle) + \
        #                     self.env_variables["angle_effect_noise_sd_c"]
        # angle_deviation = angle_deviation.item()
        # angle = angle + float(angle_deviation)

        # NEW (02.11.22 onwards)
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
