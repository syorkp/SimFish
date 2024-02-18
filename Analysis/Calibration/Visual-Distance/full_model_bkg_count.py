"""
Script to directly compute photons from background_brightness given all background_brightness, decay, arena dims, luminance.
"""

import numpy as np
from Environment.Board.drawing_board import DrawingBoard
from Environment.Fish.eye import Eye
from Analysis.load_model_config import load_assay_configuration_files


def get_max_background_brightness(background_brightness, light_decay_rate, pr_size, width, height, luminance, env_variables):
    # Set fish position in corner.
    fish_position = np.array([width/2, height/2])
    fish_orientation = 0

    max_visual_distance = np.absolute(np.log(0.001) / env_variables["light_decay_rate"])

    # Create board and get masked pixels
    board = DrawingBoard(width, height, light_decay_rate, light_decay_rate, pr_size, False, False, prey_radius=1,
                         light_gain=luminance, visible_scatter=background_brightness)
    board.erase(background_brightness)
    masked_pixels, lum_mask = board.get_masked_pixels(fish_position, np.array([]), np.array([]))

    # Create eye
    verg_angle = 77. * (np.pi / 180)
    retinal_field = 163. * (np.pi / 180)
    dark_col = int(env_variables['arena_width'] * env_variables['dark_light_ratio'])
    left_eye = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False, max_visual_range=max_visual_distance)

    # Create eye position
    left_eye_pos = (
        +np.cos(np.pi / 2 - fish_orientation) * env_variables['eyes_biasx'] + fish_position[0],
        -np.sin(np.pi / 2 - fish_orientation) * env_variables['eyes_biasx'] + fish_position[1])

    prey_locations = []
    sand_grain_locations = []

    left_eye.read(masked_pixels, left_eye_pos[0], left_eye_pos[1], fish_orientation, lum_mask, prey_locations, sand_grain_locations)

    observation = left_eye.readings
    observation = np.floor(observation).astype(int)
    uv = observation[:, 1]
    max_uv = np.max(uv)
    return max_uv


# decay = 0.01
# max_distance_s = (1500**2 + 1500**2) ** 0.5
# luminance = 200
# distance = 600
# background_brightness = 0.1
# rf_size = 0.0133 * 3
# learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_10-1")
#
# print(get_max_background_brightness(background_brightness, decay, rf_size, 1500, 1500, luminance, env_variables))