import numpy as np
import json
import matplotlib.pyplot as plt
import skimage.draw as draw
from skimage import io

from Analysis.load_data import load_data
from Configurations.Networks.original_network import base_network_layers, ops, connectivity
from Tools.make_gif import make_gif
from skimage.transform import resize, rescale


class DrawingBoard:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.db = self.get_base_arena(0.3)

    def get_base_arena(self, bkg=0.3):
        db = (np.ones((self.height, self.width, 3), dtype=np.double) * bkg)
        db[1:2, :] = np.array([1, 0, 0])
        db[self.width - 2:self.width - 1, :] = np.array([1, 0, 0])
        db[:, 1:2] = np.array([1, 0, 0])
        db[:, self.height - 2:self.height - 1] = np.array([1, 0, 0])
        return db

    def circle(self, center, rad, color):
        rr, cc = draw.circle(center[1], center[0], rad, self.db.shape)
        self.db[rr, cc, :] = color

    @staticmethod
    def multi_circles(cx, cy, rad):
        rr, cc = draw.circle(0, 0, rad)
        rrs = np.tile(rr, (len(cy), 1)) + np.tile(np.reshape(cy, (len(cy), 1)), (1, len(rr)))
        ccs = np.tile(cc, (len(cx), 1)) + np.tile(np.reshape(cx, (len(cx), 1)), (1, len(cc)))
        return rrs, ccs

    def overlay_salt(self, salt_location):
        """Show salt source"""
        # Consider modifying so that shows distribution.
        self.circle(salt_location, 20, (0, 1, 1))

    def tail(self, head, left, right, tip, color):
        tail_coordinates = np.array((head, left, tip, right))
        rr, cc = draw.polygon(tail_coordinates[:, 1], tail_coordinates[:, 0], self.db.shape)
        self.db[rr, cc, :] = color

    def fish_shape(self, mouth_centre, mouth_rad, head_rad, tail_length, mouth_colour, body_colour, angle):
        offset = np.pi / 2
        angle += offset
        angle = -angle
        self.circle(mouth_centre, mouth_rad, mouth_colour)  # For the mouth.
        dx1, dy1 = head_rad * np.sin(angle), head_rad * np.cos(angle)
        head_centre = (mouth_centre[0] + dx1,
                       mouth_centre[1] + dy1)
        self.circle(head_centre, head_rad, body_colour)
        dx2, dy2 = -1 * dy1, dx1
        left_flank = (head_centre[0] + dx2,
                      head_centre[1] + dy2)
        right_flank = (head_centre[0] - dx2,
                       head_centre[1] - dy2)
        tip = (mouth_centre[0] + (tail_length + head_rad) * np.sin(angle),
               mouth_centre[1] + (tail_length + head_rad) * np.cos(angle))
        self.tail(head_centre, left_flank, right_flank, tip, body_colour)

    def show_action_continuous(self, impulse, angle, fish_angle, x_position, y_position, colour):
        # rr, cc = draw.ellipse(int(y_position), int(x_position), (abs(angle) * 3) + 3, (impulse*0.5) + 3, rotation=-fish_angle)
        rr, cc = draw.ellipse(int(y_position), int(x_position), 3, (impulse*0.5) + 3, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def show_action_discrete(self, fish_angle, x_position, y_position, colour):
        rr, cc = draw.ellipse(int(y_position), int(x_position), 5, 3, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def get_action_colour(self, action):
        """Returns the (R, G, B) for associated actions"""
        if action == 0:  # Slow2
            action_colour = (0, 1, 0)

        elif action == 1:  # RT right
            action_colour = (0, 1, 0)

        elif action == 2:  # RT left
            action_colour = (0, 1, 0)

        elif action == 3:  # Short capture swim
            action_colour = (1, 0, 1)

        elif action == 4:  # j turn right
            action_colour = (1, 1, 1)

        elif action == 5:  # j turn left
            action_colour = (1, 1, 1)

        elif action == 6:  # Do nothing
            action_colour = (0, 0, 0)

        elif action == 7:  # c start right
            action_colour = (1, 0, 0)

        elif action == 8:  # c start left
            action_colour = (1, 0, 0)

        elif action == 9:  # Approach swim.
            action_colour = (0, 1, 0)

        else:
            action_colour = (0, 0, 0)
            print("Invalid action given")

        return action_colour


def draw_previous_actions(board, past_actions, past_positions, fish_angles, continuous_actions=True, n_actions_to_show=50):
    while len(past_actions) > n_actions_to_show:
        past_actions.pop(0)
    while len(past_positions) > n_actions_to_show:
        past_positions.pop(0)
    while len(fish_angles) > n_actions_to_show:
        fish_angles.pop(0)

    for i, a in enumerate(past_actions):
        if continuous_actions:
            action_colour = (1 * ((i + 1) / len(past_actions)), 0, 0)
            board.show_action_continuous(a[0], a[1], fish_angles[i], past_positions[i][0],
                                              past_positions[i][1], action_colour)
        else:
            action_colour = board.get_action_colour(fish_angles[i])
            board.show_action_discrete(fish_angles[i], past_positions[i][0],
                                            past_positions[i][1], action_colour)
    return board, past_actions, past_positions, fish_angles


def draw_episode(data, config_name, model_name, draw_past_actions=True, show_energy_state=True, scale=0.25):
    with open(f"../../Configurations/Training-Configs/{config_name}/1_env.json", 'r') as f:
        env_variables = json.load(f)

    n_actions_to_show = 50
    board = DrawingBoard(1500, 1500)
    energy_levels = data["internal_state"][:, 0]
    fish_positions = data["fish_position"]
    num_steps = fish_positions.shape[0]
    frames = []
    action_buffer = []
    position_buffer = []
    orientation_buffer = []

    for step in range(num_steps):
        action_buffer.append([data["impulse"][step], data["angle"][step]])
        position_buffer.append(fish_positions[step])
        orientation_buffer.append(data["fish_angle"][step])

        if draw_past_actions:
            board, action_buffer, position_buffer, orientation_buffer = draw_previous_actions(board, action_buffer, position_buffer, orientation_buffer)

        if show_energy_state:
            fish_body_colour = (1-energy_levels[step], energy_levels[step], 0)
        else:
            fish_body_colour = (0, 1, 0)

        board.fish_shape(fish_positions[step], env_variables['fish_mouth_size'],
                              env_variables['fish_head_size'], env_variables['fish_tail_length'],
                         (0, 1, 0), fish_body_colour, data["fish_angle"][step])

        px = np.round(np.array([pr[0] for pr in data["prey_positions"][step]])).astype(int)
        py = np.round(np.array([pr[1] for pr in data["prey_positions"][step]])).astype(int)
        rrs, ccs = board.multi_circles(px, py, env_variables["prey_size_visualisation"])

        rrs = np.clip(rrs, 0, 1499)
        ccs = np.clip(ccs, 0, 1499)

        board.db[rrs, ccs] = (0, 0, 1)

        if data["predator_presence"][step]:
            board.circle(data["predator_positions"][step], env_variables['predator_size'], (0, 1, 0))

        frames.append(rescale(board.db, scale, multichannel=True, anti_aliasing=True))
        board.db = board.get_base_arena(0.3)

    frames = np.array(frames)
    frames *= 255
    make_gif(frames, f"{model_name}-4-behaviour.gif", duration=len(frames) * 0.03, true_image=True)


# model_name = "parameterised_speed_test_fast-1"
model_name = "scaffold_version_4-4"
data = load_data(model_name, "Behavioural-Data-Free", "Naturalistic-4")
config_name = "ppo_continuous_sbe_is_scaffold_4"

draw_episode(data, config_name, model_name)
