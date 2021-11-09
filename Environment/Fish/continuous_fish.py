from Environment.Fish.fish import Fish


class ContinuousFish(Fish):

    def __init__(self, board, env_variables, dark_col, realistic_bouts, new_simulation, fish_mass=None):
        super().__init__(board, env_variables, dark_col, realistic_bouts, new_simulation, fish_mass)

        self.making_capture = True

    def calculate_distance(self, impulse):
        return (1.771548 * impulse + self.env_variables['fish_mass'] * 0.004644 + 0.081417)/10

    def take_action(self, action):
        impulse = action[0]
        angle = action[1]
        distance = self.calculate_distance(impulse)
        reward = - self.calculate_action_cost(angle, distance) - self.env_variables['baseline_penalty']
        self.prev_action_impulse = impulse
        self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
        self.body.angle += angle
        return reward
