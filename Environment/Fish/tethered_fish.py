from Environment.Fish.fish import Fish


class TetheredFish(Fish):

    """
    Same as normal fish, though overwrites any movement consequences of action choice.
    """

    def __init__(self, board, env_variables, dark_col, using_gpu):
        super().__init__(board, env_variables, dark_col, using_gpu)

    def take_action(self, action):
        if action == 0:  # Swim forward
            self.head.color = (0, 1, 0)
        elif action == 1:  # Turn right
            self.head.color = (0, 1, 0)
        elif action == 2:   # Turn left
            self.head.color = (0, 1, 0)
        elif action == 3:   # Capture
            self.head.color = [1, 0, 1]
            self.making_capture = True
        elif action == 4:  # j turn right
            self.head.color = [1, 1, 1]
        elif action == 5:  # j turn left
            self.head.color = [1, 1, 1]
        elif action == 6:   # do nothing:
            reward = 0
        elif action == 7:   # c start right
            reward = 0
        elif action == 8:   # c start left
            reward = 0
        elif action == 9:   # approach swim
            reward = 0
        elif action == 10:  # approach swim
            reward = 0
        elif action == 11:  # approach swim
            reward = 0
        else:
            reward = None
            print("Invalid action given")

        return 0
