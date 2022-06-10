import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_configuration_files
from Environment.Fish.eye import Eye

# from Analysis.Video.behaviour_video_construction import DrawingBoard
from Tools.drawing_board_new import NewDrawingBoard
from Analysis.load_data import load_data



def visualise_environent_at_step(model_name, assay_config, assay_id, step_to_draw):
    data = load_data(model_name, assay_config, assay_id)
    learning_params, env_variables, n, b, c = load_configuration_files(model_name="dqn_scaffold_18-1")
    fish_body_colour = (0, 1, 0)
    board = NewDrawingBoard(env_variables["width"], env_variables["height"], env_variables["decay_rate"],
                            env_variables["uv_photoreceptor_rf_size"],
                            using_gpu=False, visualise_mask=False, prey_size=1,
                            light_gain=env_variables["light_gain"], visible_scatter=env_variables["bkg_scatter"],
                            light_gradient=env_variables["light_gradient"],
                            dark_light_ratio=env_variables['dark_light_ratio'],
                            red_occlusion_gain=env_variables["red_occlusion_gain"],
                            uv_occlusion_gain=env_variables["uv_occlusion_gain"],
                            red2_occlusion_gain=env_variables["red2_occlusion_gain"])

    # Draw shapes for visualisation
    board.erase_visualisation(0.3)
    board.fish_shape(data["fish_position"][step_to_draw], env_variables['fish_mouth_size'],
                     env_variables['fish_head_size'], env_variables['fish_tail_length'],
                     (0, 1, 0), fish_body_colour, data["fish_angle"][step_to_draw])

    px = np.round(np.array([pr[0] for pr in data["prey_positions"][step_to_draw]])).astype(int)
    py = np.round(np.array([pr[1] for pr in data["prey_positions"][step_to_draw]])).astype(int)
    rrs, ccs = board.multi_circles(px, py, env_variables["prey_size_visualisation"])

    rrs = np.clip(rrs, 0, 1499)
    ccs = np.clip(ccs, 0, 1499)

    board.db_visualisation[rrs, ccs] = (0, 0, 1)

    if data["predator_presence"][step_to_draw]:
        board.circle(data["predator_positions"][step_to_draw], env_variables['predator_size'], (0, 1, 0))

    relative_dark_gain = env_variables["dark_gain"] / env_variables["light_gain"]
    board.apply_light(int(env_variables['width'] * env_variables['dark_light_ratio']), relative_dark_gain, 1, visualisation=True)

    # Make red walls thicker and more visible.
    board.db_visualisation[0:10, :] = [1, 0, 0]
    board.db_visualisation[env_variables["width"] - 10:, :] = [1, 0, 0]
    board.db_visualisation[:, :10] = [1, 0, 0]
    board.db_visualisation[:, env_variables["height"] - 10:] = [1, 0, 0]


    # Draw shapes for image
    board.erase(env_variables['bkg_scatter'])
    board.fish_shape(data["fish_position"][step_to_draw], env_variables['fish_mouth_size'],
                     env_variables['fish_head_size'], env_variables['fish_tail_length'],
                     (0, 1, 0), fish_body_colour, data["fish_angle"][step_to_draw])

    px = np.round(np.array([pr[0] for pr in data["prey_positions"][step_to_draw]])).astype(int)
    py = np.round(np.array([pr[1] for pr in data["prey_positions"][step_to_draw]])).astype(int)
    rrs, ccs = board.multi_circles(px, py, env_variables["prey_size"])

    rrs = np.clip(rrs, 0, 1499)
    ccs = np.clip(ccs, 0, 1499)

    board.db[rrs, ccs] = (0, 0, 1)


    # Show visualisation

    plt.figure(figsize=(10, 10))
    plt.imshow(board.db_visualisation)
    plt.show()

    # Also create image of zoomed in of fish
    buffer = 200
    surrounding_fish_region = board.db_visualisation[int(data["fish_position"][step_to_draw, 1]-buffer):
                                                     int(data["fish_position"][step_to_draw, 1]+buffer),
                                                     int(data["fish_position"][step_to_draw, 0] - buffer):
                                                     int(data["fish_position"][step_to_draw, 0] + buffer),
                                                    ]
    plt.imshow(surrounding_fish_region)
    plt.show()


    # Print version with PR occupancy also shown.
    dark_col = int(env_variables['width'] * env_variables['dark_light_ratio'])
    verg_angle = env_variables['eyes_verg_angle'] * (np.pi / 180)
    retinal_field = env_variables['visual_field'] * (np.pi / 180)
    test_eye_l = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False)
    test_eye_r = Eye(board, verg_angle, retinal_field, False, env_variables, dark_col, False)

    fish_angle = data["fish_angle"][step_to_draw]
    fish_position = data["fish_position"][step_to_draw]

    right_eye_pos = (
        -np.cos(np.pi / 2 - fish_angle) * env_variables['eyes_biasx'] + data["fish_position"][step_to_draw, 0],
        +np.sin(np.pi / 2 - fish_angle) * env_variables['eyes_biasx'] + data["fish_position"][step_to_draw, 1])
    left_eye_pos = (
        +np.cos(np.pi / 2 - fish_angle) * env_variables['eyes_biasx'] + data["fish_position"][step_to_draw, 0],
        -np.sin(np.pi / 2 - fish_angle) * env_variables['eyes_biasx'] + data["fish_position"][step_to_draw, 1])

    arena = np.zeros((1500, 1500))

    predator_bodies = np.array([[fish_position[0]-100, fish_position[1]+75]])
    full_masked_image = board.get_masked_pixels(np.array([data["fish_position"][step_to_draw][0], data["fish_position"][step_to_draw][1]]),
                                                data["prey_positions"][step_to_draw],
                                                predator_bodies
                                                )

    # Left eye
    channel_angles_surrounding = test_eye_l.channel_angles_surrounding_stacked + fish_angle
    uv_arena_pixels = full_masked_image[:, :, 1:2]
    red_arena_pixels = np.concatenate(
         (full_masked_image[:, :, 0:1], full_masked_image[:, :, 2:]), axis=2)
    uv_points, red_points = test_eye_l.get_pr_coverage(masked_arena_pixels_uv=uv_arena_pixels,
                                                       masked_arena_pixels_red=red_arena_pixels,
                                                       eye_x=left_eye_pos[0],
                                                       eye_y=left_eye_pos[1],
                                                       channel_angles_surrounding=channel_angles_surrounding,
                                                       n_channels_uv=test_eye_l.uv_photoreceptor_num,
                                                       n_channels_red=test_eye_l.red_photoreceptor_num)

    uv_points = np.unique(np.reshape(uv_points, (-1, 2)), axis=0)
    arena[uv_points[:, 1], uv_points[:, 0]] = 1
    board.db_visualisation[uv_points[:, 1], uv_points[:, 0], 0:2] += 0.5

    # Right eye
    channel_angles_surrounding = test_eye_r.channel_angles_surrounding_stacked + fish_angle
    uv_arena_pixels = full_masked_image[:, :, 1:2]
    red_arena_pixels = np.concatenate(
         (full_masked_image[:, :, 0:1], full_masked_image[:, :, 2:]), axis=2)
    uv_points, red_points = test_eye_r.get_pr_coverage(masked_arena_pixels_uv=uv_arena_pixels,
                                                       masked_arena_pixels_red=red_arena_pixels,
                                                       eye_x=right_eye_pos[0],
                                                       eye_y=right_eye_pos[1],
                                                       channel_angles_surrounding=channel_angles_surrounding,
                                                       n_channels_uv=test_eye_r.uv_photoreceptor_num,
                                                       n_channels_red=test_eye_r.red_photoreceptor_num)

    uv_points = np.unique(np.reshape(uv_points, (-1, 2)), axis=0)
    arena[uv_points[:, 1], uv_points[:, 0]] = 1

    plt.figure(figsize=(15, 15), dpi=100)
    plt.imshow(arena)
    plt.show()

    board.db_visualisation[uv_points[:, 1], uv_points[:, 0], 0:2] += 0.3
    plt.figure(figsize=(15, 15), dpi=100)
    plt.imshow(board.db_visualisation)
    plt.show()

    # Get observation for the given step.
    test_eye_l.read(full_masked_image, left_eye_pos[0], left_eye_pos[1], fish_angle)
    test_eye_r.read(full_masked_image, right_eye_pos[0], right_eye_pos[1], fish_angle)


    photons_l = np.floor(test_eye_l.readings).astype(int)
    photons_l = np.concatenate((photons_l[:, 0:1], photons_l[:, 2:3], photons_l[:, 1:2]), axis=1)
    photons_l = photons_l.clip(0, 255)

    photons_r = np.floor(test_eye_r.readings).astype(int)
    photons_r = np.concatenate((photons_r[:, 0:1], photons_r[:, 2:3], photons_r[:, 1:2]), axis=1)
    photons_r = photons_r.clip(0, 255)

    photons_l = np.expand_dims(photons_l, 0)
    photons_l = np.repeat(photons_l, 20, 0)
    photons_r = np.expand_dims(photons_r, 0)
    photons_r = np.repeat(photons_r, 20, 0)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(photons_l)
    axs[1].imshow(photons_r)
    plt.show()

    # Show background
    plt.imshow(board.background_grating)
    plt.show()


    # TODO: Print scatter, obstrution, background, and light gain masks in parallel.
    AB, L, O, S = board.get_masked_pixels(np.array([data["fish_position"][step_to_draw][0], data["fish_position"][step_to_draw][1]]),
                                                data["prey_positions"][step_to_draw],
                                                predator_bodies, return_masks=True)
    # Rearrange AB
    plt.imshow(AB)
    plt.show()

    plt.imshow(L)
    plt.show()

    plt.imshow(O)
    plt.show()

    plt.imshow(S)
    plt.show()


    # TODO: Get rid of ticks and instead show scale bar.


visualise_environent_at_step("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-3", 54)




