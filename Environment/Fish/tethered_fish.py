from Environment.Fish.fish import Fish


class TetheredFish(Fish):

    """
    Same as normal fish, though overwrites any movement consequences of action choice.
    """

    def __init__(self, board, env_variables, dark_col, realistic_bouts):
        super().__init__(board, env_variables, dark_col, realistic_bouts)

    def take_action(self, action):
        if action == 0:  # Swim forward
            reward = -self.env_variables['forward_swim_cost']
            self.head.color = (0, 1, 0)
        elif action == 1:  # Turn right
            reward = -self.env_variables['routine_turn_cost']
            self.head.color = (0, 1, 0)
        elif action == 2:   # Turn left
            reward = -self.env_variables['routine_turn_cost']
            self.head.color = (0, 1, 0)
        elif action == 3:   # Capture
            reward = -self.env_variables['capture_swim_cost']
            self.head.color = [1, 0, 1]
            self.making_capture = True
        elif action == 4:  # j turn right
            reward = -self.env_variables['j_turn_cost']
            self.head.color = [1, 1, 1]
        elif action == 5:  # j turn left
            reward = -self.env_variables['j_turn_cost']
            self.head.color = [1, 1, 1]
        elif action == 6:   # do nothing:
            reward = -self.env_variables['rest_cost']
        else:
            reward = None
            print("Invalid action given")

        # elif action == 6: #converge eyes. Be sure to update below with fish.[]
        #     self.verg_angle = 77 * (np.pi / 180)
        #     self.left_eye.update_angles(self.verg_angle, self.retinal_field, True)
        #     self.right_eye.update_angles(self.verg_angle, self.retinal_field, False)
        #     self.conv_state = 1

        # elif action == 7: #diverge eyes
        #     self.verg_angle = 25 * (np.pi / 180)
        #     self.left_eye.update_angles(self.verg_angle, self.retinal_field, True)
        #     self.right_eye.update_angles(self.verg_angle, self.retinal_field, False)
        #     self.conv_state = 0
        return reward
