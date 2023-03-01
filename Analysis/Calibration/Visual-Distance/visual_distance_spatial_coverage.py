"""
A way to display the PR coverage around the fish.
"""
import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_assay_configuration_files
from Environment.Fish.eye import Eye
from Tools.drawing_board import DrawingBoard


def display_pr_coverage(model_name):
    learning_params, env_variables, n, b, c = load_assay_configuration_files(model_name)
    max_photoreceptor_rf_size = max([env_variables['uv_photoreceptor_rf_size'],
                                     env_variables['red_photoreceptor_rf_size']])

    board = DrawingBoard(env_variables['width'], env_variables['height'],
                         uv_light_decay_rate=env_variables['light_decay_rate'],
                         red_light_decay_rate=env_variables['light_decay_rate'],
                         photoreceptor_rf_size=max_photoreceptor_rf_size,
                         using_gpu=False,
                         prey_radius=env_variables['prey_radius'],
                         predator_radius=env_variables['predator_radius'],
                         visible_scatter=env_variables['background_brightness'],
                         dark_light_ratio=env_variables['dark_light_ratio'],
                         dark_gain=env_variables['dark_gain'],
                         light_gain=env_variables['light_gain'],
                         )

    dark_col = int(env_variables['width'] * env_variables['dark_light_ratio'])
    verg_angle = env_variables['eyes_verg_angle'] * (np.pi / 180)
    retinal_field = env_variables['visual_field'] * (np.pi / 180)
    test_eye_l = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False, 700)
    test_eye_r = Eye(board, verg_angle, retinal_field, False, env_variables, dark_col, False, 700)

    right_eye_pos = (
        -np.cos(np.pi / 2 - 0) * env_variables['eyes_biasx'] + 500,
        +np.sin(np.pi / 2 - 0) * env_variables['eyes_biasx'] + 500)
    left_eye_pos = (
        +np.cos(np.pi / 2 - 0) * env_variables['eyes_biasx'] + 500,
        -np.sin(np.pi / 2 - 0) * env_variables['eyes_biasx'] + 500)

    arena = np.zeros((1500, 1500))


    # self.show_new_channel_sectors(left_eye_pos, right_eye_pos)  # TODO: Needs to be updated.
    predator_bodies = np.array([])
    full_masked_image = board.get_masked_pixels(np.array([500, 500]),
                                                     np.array([[750, 750], [500, 550]]),
                                                     predator_bodies
                                                     )

    # Left eye
    channel_angles_surrounding = test_eye_l.photoreceptor_angles_surrounding_stacked + 0
    uv_arena_pixels = full_masked_image[:, :, 1:2]
    red_arena_pixels = np.concatenate(
         (full_masked_image[:, :, 0:1], full_masked_image[:, :, 2:]), axis=2)
    uv_points, red_points = test_eye_l.get_pr_coverage(masked_arena_pixels_uv=uv_arena_pixels,
                                                       masked_arena_pixels_red=red_arena_pixels,
                                                       eye_x=left_eye_pos[0],
                                                       eye_y=left_eye_pos[1],
                                                       photoreceptor_angles_surrounding=channel_angles_surrounding,
                                                       n_photoreceptors_uv=test_eye_l.uv_photoreceptor_num,
                                                       n_photoreceptors_red=test_eye_l.red_photoreceptor_num)

    uv_points = np.unique(np.reshape(uv_points, (-1, 2)), axis=0)
    arena[uv_points[:, 0], uv_points[:, 1]] = 1

    # Right eye
    channel_angles_surrounding = test_eye_r.photoreceptor_angles_surrounding_stacked + 0
    uv_arena_pixels = full_masked_image[:, :, 1:2]
    red_arena_pixels = np.concatenate(
         (full_masked_image[:, :, 0:1], full_masked_image[:, :, 2:]), axis=2)
    uv_points, red_points = test_eye_r.get_pr_coverage(masked_arena_pixels_uv=uv_arena_pixels,
                                                       masked_arena_pixels_red=red_arena_pixels,
                                                       eye_x=right_eye_pos[0],
                                                       eye_y=right_eye_pos[1],
                                                       photoreceptor_angles_surrounding=channel_angles_surrounding,
                                                       n_photoreceptors_uv=test_eye_r.uv_photoreceptor_num,
                                                       n_photoreceptors_red=test_eye_r.red_photoreceptor_num)

    uv_points = np.unique(np.reshape(uv_points, (-1, 2)), axis=0)
    arena[uv_points[:, 0], uv_points[:, 1]] = 1

    plt.figure(figsize=(15, 15), dpi=100)
    plt.scatter([500], [500], color="r")
    plt.imshow(arena)
    plt.show()


if __name__ == "__main__":
    model_name = "dqn_gamma-1"
    display_pr_coverage(model_name)
