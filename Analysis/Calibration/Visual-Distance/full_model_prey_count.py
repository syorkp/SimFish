"""
Script to directly compute photons from bkg_scatter given all bkg_scatter, decay, arena dims, luminance.
"""

import numpy as np
from Tools.drawing_board_new import NewDrawingBoard
from Environment.Fish.eye import Eye
from Analysis.load_model_config import load_configuration_files


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def full_model_prey_count(bkg_scatter, decay_rate, pr_size, width, height, luminance, env_variables, prey_distance):
    # Set fish position in corner.
    fish_position = np.array([20, 20])
    fish_orientation = np.pi/4

    # Create board and get masked pixels
    board = NewDrawingBoard(width, height, decay_rate, pr_size, False, False, 1, light_gain=luminance, visible_scatter=bkg_scatter)
    board.erase(bkg_scatter)
    masked_pixels = board.get_masked_pixels(fish_position, np.array([]), np.array([]))

    # Create eye
    verg_angle = 77. * (np.pi / 180)
    retinal_field = 163. * (np.pi / 180)
    dark_col = int(env_variables['width'] * env_variables['dark_light_ratio'])
    left_eye = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False)

    # Create eye positions

    left_eye_pos = (
        +np.cos(np.pi / 2 - fish_orientation) * env_variables['eyes_biasx'] + fish_position[0],
        -np.sin(np.pi / 2 - fish_orientation) * env_variables['eyes_biasx'] + fish_position[1])

    # channel_angles_surrounding = left_eye.channel_angles_surrounding_stacked
    # uv_ph_coords_l, red_ph_coords_l = left_eye.get_pr_coverage(masked_pixels[:, :, 1],
    #                                           np.concatenate((masked_pixels[:, :, :1], masked_pixels[:, :, 2:]), axis=2),
    #                                           left_eye_pos[0], left_eye_pos[1], channel_angles_surrounding, env_variables["uv_photoreceptor_num"],
    #                                           env_variables["red_photoreceptor_num"])
    #
    # uv_ph_coords_l = np.reshape(uv_ph_coords_l, (uv_ph_coords_l.shape[0] * uv_ph_coords_l.shape[1], 2))
    # relative_coords = uv_ph_coords_l - fish_position
    # unviable = (relative_coords[:, 0] == 0) * 1
    # unviable += (relative_coords[:, 1] == 0) * 1
    # relative_coords = relative_coords[unviable == 0]
    # uv_covered_distances = (relative_coords[:, 0] ** 2 + relative_coords[:, 1] ** 2) ** 0.5
    # closest = find_nearest(uv_covered_distances, prey_distance)
    # prey_location = uv_ph_coords_l[closest, :]
    # field[uv_ph_coords_l[:, 0], uv_ph_coords_l[:, 1]] = 1

    coords = left_eye.get_pr_line_coordinates_uv(left_eye_pos[0], left_eye_pos[1])
    coords = coords[-1, :, :]
    relative_coords = coords - fish_position
    uv_covered_distances = (relative_coords[:, 0] ** 2 + relative_coords[:, 1] ** 2) ** 0.5
    closest = find_nearest(uv_covered_distances, prey_distance)
    prey_location = coords[closest, :]

    # field = np.zeros((1500, 1500))
    # field[coords[:, 0], coords[:, 1]] = 1

    board.erase(bkg_scatter)
    px = np.round(np.array([prey_location[0]])).astype(int)
    py = np.round(np.array([prey_location[1]])).astype(int)
    rrs, ccs = board.multi_circles(px, py, 1)
    rrs = np.clip(rrs, 0, 1499)
    ccs = np.clip(ccs, 0, 1499)
    board.db[rrs, ccs] = (0, 0, 1)

    masked_pixels = board.get_masked_pixels(fish_position, np.array([prey_location]), np.array([]))
    left_eye.read(masked_pixels, left_eye_pos[0], left_eye_pos[1], fish_orientation)

    observation = left_eye.readings  # np.dstack((left_eye.readings, right_eye.readings))
    observation = np.floor(observation).astype(int)
    uv = observation[:, 1]
    max_uv = np.max(uv)
    return max_uv


# decay = 0.01
# max_distance_s = (1500**2 + 1500**2) ** 0.5
# luminance = 200
# distance = 600
# bkg_scatter = 0.1
# rf_size = 0.0133 * 3
# learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_10-1")
# prey_distance = 100
#
# print(full_model_prey_count(bkg_scatter, decay, rf_size, 1500, 1500, luminance, env_variables, prey_distance))