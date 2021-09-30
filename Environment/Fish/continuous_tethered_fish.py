from Environment.Fish.fish import Fish


class ContinuousTetheredFish(Fish):

    """
    Same as normal fish, though overwrites any movement consequences of action choice.
    """

    def __init__(self, board, env_variables, dark_col, realistic_bouts):
        super().__init__(board, env_variables, dark_col, realistic_bouts)

    def calculate_distance(self, impulse):
        return (1.771548 * impulse + self.env_variables['fish_mass'] * 0.004644 + 0.081417)/10

    def take_action(self, action):
        impulse = action[0]
        angle = action[1]
        distance = self.calculate_distance(impulse)
        reward = - self.calculate_action_cost(angle, distance) - self.env_variables['baseline_penalty']
        return reward
