from Environment.Fish.continuous_fish import ContinuousFish


class ContinuousTetheredFish(ContinuousFish):

    """
    Same as normal fish, though overwrites any movement consequences of action choice.
    """

    def __init__(self, board, env_variables, dark_col):
        super().__init__(board, env_variables, dark_col)

    def take_action(self, action):
        return 0
