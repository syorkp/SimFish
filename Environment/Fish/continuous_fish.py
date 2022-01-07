from Environment.Fish.fish import Fish


class ContinuousFish(Fish):

    def __init__(self, board, env_variables, dark_col, realistic_bouts, new_simulation, using_gpu, fish_mass=None):
        super().__init__(board, env_variables, dark_col, realistic_bouts, new_simulation, using_gpu, fish_mass=fish_mass)

        self.making_capture = True
        self.new_simulation = new_simulation

    def calculate_distance(self, impulse):
        return (1.771548 * impulse + self.env_variables['fish_mass'] * 0.004644 + 0.081417)/10

    def take_action(self, action):
        """Overrides method for discrete fish"""
        if self.new_simulation:
            return self._take_action_new(action)
        else:
            return self._take_action(action)

    def _take_action(self, action):
        impulse = action[0]
        angle = action[1]
        distance = self.calculate_distance(impulse)
        reward = - self.calculate_action_cost(angle, distance) - self.env_variables['baseline_penalty']
        self.prev_action_impulse = impulse
        self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
        self.body.angle += angle
        return reward

    def _take_action_new(self, action):
        impulse = action[0]
        angle = action[1]
        self.prev_action_impulse = impulse
        self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
        self.body.angle += angle
        return 0.0

    def update_energy_level(self, reward, impulse, angle, consumption):
        """Updates the current energy state for continuous fish, overrides in fish class."""
        intake_s = self.intake_scale(self.energy_level)
        action_s = self.action_scale(self.energy_level)

        consumption = 1.0 * consumption  # TODO: Make sure this works

        energy_intake = (intake_s * consumption * self.cc)
        energy_use = (action_s * (self.ci * impulse + self.ca * angle + self.baseline_decrease))
        self.energy_level += energy_intake - energy_use
        reward += (energy_intake * self.consumption_reward_scaling) - (energy_use * self.action_reward_scaling)
        return reward
