import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from Analysis.load_model_config import load_assay_configuration_files
from Environment.Fish.eye import Eye

# from Analysis.Video.behaviour_video_construction import DrawingBoard
from Tools.drawing_board import DrawingBoard
from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.anchored_scale_bar import AnchoredHScaleBar


def visualise_environent_at_step(model_name, assay_config, assay_id, step_to_draw, reduction=2.5):
    data = load_data(model_name, assay_config, assay_id)
    learning_params, env_variables, n, b, c = load_assay_configuration_files(model_name="dqn_scaffold_18-1")
    env_variables["arena_width"] = int(env_variables["arena_width"]/reduction)
    env_variables["arena_height"] = int(env_variables["arena_height"]/reduction)
    env_variables["red_photoreceptor_num"] = int(env_variables["red_photoreceptor_num"]/reduction)
    env_variables["uv_photoreceptor_num"] = int(env_variables["uv_photoreceptor_num"]/reduction)

    env_variables["light_gradient"] = int(env_variables["light_gradient"]/reduction)
    fish_body_colour = (0, 1, 0)
    board = DrawingBoard(env_variables["arena_width"], env_variables["arena_height"], env_variables["light_decay_rate"],
                            env_variables["uv_photoreceptor_rf_size"],
                            using_gpu=False, prey_radius=1,
                            light_gain=env_variables["light_gain"], visible_scatter=env_variables["background_brightness"],
                            light_gradient=env_variables["light_gradient"],
                            dark_light_ratio=env_variables['dark_light_ratio'],
)


    fish_angle = data["fish_angle"][step_to_draw]
    fish_position = data["fish_position"][step_to_draw]/reduction
    predator_bodies = np.array([[fish_position[0]+600/reduction, fish_position[1]-400/reduction]])

    # Draw shapes for visualisation
    board.erase_visualisation(0.3)
    board.fish_shape(fish_position, env_variables['fish_mouth_radius'],
                     env_variables['fish_head_radius'], env_variables['fish_tail_length'],
                     (0, 1, 0), fish_body_colour, data["fish_angle"][step_to_draw])

    px = np.round(np.array([pr[0]/reduction for pr in data["prey_positions"][step_to_draw]])).astype(int)
    py = np.round(np.array([pr[1]/reduction for pr in data["prey_positions"][step_to_draw]])).astype(int)
    rrs, ccs = board.multi_circles(px, py, env_variables["prey_radius_visualisation"])

    rrs = np.clip(rrs, 0, env_variables["arena_width"]-1)
    ccs = np.clip(ccs, 0, env_variables["arena_height"]-1)

    board.db_visualisation[rrs, ccs] = (0, 0, 1)

    relative_dark_gain = env_variables["dark_gain"] / env_variables["light_gain"]
    board.apply_light(int(env_variables['arena_width'] * env_variables['dark_light_ratio']), relative_dark_gain, 1, visualisation=True)

    # Make red walls thicker and more visible.
    board.db_visualisation[0:10, :] = [1, 0, 0]
    board.db_visualisation[env_variables["arena_width"] - 10:, :] = [1, 0, 0]
    board.db_visualisation[:, :10] = [1, 0, 0]
    board.db_visualisation[:, env_variables["arena_height"] - 10:] = [1, 0, 0]
    board.circle(predator_bodies[0], env_variables['predator_radius'], (0, 1, 0), visualisation=True)

    # Draw shapes for image
    board.erase(env_variables['background_brightness'])
    board.fish_shape(fish_position, env_variables['fish_mouth_radius'],
                     env_variables['fish_head_radius'], env_variables['fish_tail_length'],
                     (0, 1, 0), fish_body_colour, data["fish_angle"][step_to_draw])

    px = np.round(np.array([pr[0]/reduction for pr in data["prey_positions"][step_to_draw]])).astype(int)
    py = np.round(np.array([pr[1]/reduction for pr in data["prey_positions"][step_to_draw]])).astype(int)
    rrs, ccs = board.multi_circles(px, py, env_variables["prey_radius"])

    rrs = np.clip(rrs, 0, env_variables["arena_width"]-1)
    ccs = np.clip(ccs, 0, env_variables["arena_height"]-1)

    board.db[rrs, ccs] = (0, 0, 1)


    # Show visualisation of whole environment
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    plt.imshow(board.db_visualisation)
    scale_bar = AnchoredSizeBar(ax.transData,
                                200, '20mm', "lower center",
                                pad=2,
                                color='white',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 30})

    ax.add_artist(scale_bar)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig("./Panels/Panel-1/full_arena.jpg")

    plt.show()

    # Also create image of zoomed in of fish
    buffer = 300/reduction
    surrounding_fish_region = board.db_visualisation[int(fish_position[1]-buffer):
                                                     int(fish_position[1]+buffer),
                                                     int(fish_position[0]-buffer):
                                                     int(fish_position[0]+buffer),
                                                    ]
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    plt.imshow(surrounding_fish_region)
    scale_bar = AnchoredSizeBar(ax.transData,
                                100, '10mm', "lower center",
                                pad=2,
                                color='white',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 30})
    ax.add_artist(scale_bar)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig("./Panels/Panel-1/surrounding_fish_region.jpg")
    plt.show()


    # Print version with PR occupancy also shown.
    dark_col = int(env_variables['arena_width'] * env_variables['dark_light_ratio'])
    verg_angle = env_variables['eyes_verg_angle'] * (np.pi / 180)
    retinal_field = env_variables['visual_field'] * (np.pi / 180)
    test_eye_l = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False)
    test_eye_r = Eye(board, verg_angle, retinal_field, False, env_variables, dark_col, False)

    right_eye_pos = (
        -np.cos(np.pi / 2 - fish_angle) * env_variables['eyes_biasx'] + fish_position[0],
        +np.sin(np.pi / 2 - fish_angle) * env_variables['eyes_biasx'] + fish_position[1])
    left_eye_pos = (
        +np.cos(np.pi / 2 - fish_angle) * env_variables['eyes_biasx'] + fish_position[0],
        -np.sin(np.pi / 2 - fish_angle) * env_variables['eyes_biasx'] + fish_position[1])

    arena = np.zeros((env_variables["arena_width"], env_variables["arena_height"]))

    full_masked_image = board.get_masked_pixels(np.array([fish_position[0], fish_position[1]]),
                                                data["prey_positions"][step_to_draw]/reduction,
                                                predator_bodies
                                                )

    # Left eye
    channel_angles_surrounding = test_eye_l.photoreceptor_angles_surrounding_stacked + fish_angle
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
    arena[uv_points[:, 1], uv_points[:, 0]] = 1
    board.db_visualisation[uv_points[:, 1], uv_points[:, 0], 0:2] += 0.5

    # Right eye
    channel_angles_surrounding = test_eye_r.photoreceptor_angles_surrounding_stacked + fish_angle
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
    arena[uv_points[:, 1], uv_points[:, 0]] = 1

    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    plt.imshow(arena)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.show()

    board.db_visualisation[uv_points[:, 1], uv_points[:, 0], 0:2] += 0.3
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    plt.imshow(board.db_visualisation)
    scale_bar = AnchoredSizeBar(ax.transData,
                                200, '20mm', "lower center",
                                pad=2,
                                color='white',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 30})
    ax.add_artist(scale_bar)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig("./Panels/Panel-1/arena_with_channels.jpg")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    plt.imshow(full_masked_image)
    scale_bar = AnchoredSizeBar(ax.transData,
                                200, '20mm', "lower center",
                                pad=2,
                                color='white',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 30})
    ax.add_artist(scale_bar)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig("./Panels/Panel-1/full_masked_image_with_channels.jpg")
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

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(photons_l)
    axs[1].imshow(photons_r)
    axs[0].axes.get_yaxis().set_visible(False)
    axs[1].axes.get_yaxis().set_visible(False)
    axs[0].axes.get_xaxis().set_visible(False)
    axs[1].axes.get_xaxis().set_visible(False)
    plt.savefig("./Panels/Panel-1/observation.jpg")
    plt.show()

    photons_l_uv_highlight = photons_l * np.array([[1, 1, 3]])
    photons_r_uv_highlight = photons_r * np.array([[1, 1, 3]])

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(photons_l_uv_highlight)
    axs[1].imshow(photons_r_uv_highlight)
    axs[0].axes.get_yaxis().set_visible(False)
    axs[1].axes.get_yaxis().set_visible(False)
    axs[0].axes.get_xaxis().set_visible(False)
    axs[1].axes.get_xaxis().set_visible(False)
    plt.savefig("./Panels/Panel-1/observation_uv_highlight.jpg")
    plt.show()

    # Show background
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.imshow(board.background_grating)
    plt.savefig("./Panels/Panel-1/background.jpg")
    plt.show()


    # TODO: Print scatter, obstrution, background, and light gain masks in parallel.
    AB, L, O, S = board.get_masked_pixels(np.array([data["fish_position"][step_to_draw][0], data["fish_position"][step_to_draw][1]]),
                                                data["prey_positions"][step_to_draw],
                                                predator_bodies, return_masks=True)
    # Rearrange AB
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.imshow(AB)
    plt.savefig("./Panels/Panel-1/AB.jpg")
    plt.show()

    L = np.swapaxes(L, 0, 1)
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.imshow(L)
    plt.savefig("./Panels/Panel-1/L.jpg")
    plt.show()

    O = O[:, :, 0]
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.imshow(O)
    plt.savefig("./Panels/Panel-1/O.jpg")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.imshow(S)
    plt.savefig("./Panels/Panel-1/S.jpg")
    plt.show()

    # Get observation for the given step without any shot noise
    board.erase(0)
    board.fish_shape(fish_position, env_variables['fish_mouth_radius'],
                     env_variables['fish_head_radius'], env_variables['fish_tail_length'],
                     (0, 1, 0), fish_body_colour, data["fish_angle"][step_to_draw])

    px = np.round(np.array([pr[0]/reduction for pr in data["prey_positions"][step_to_draw]])).astype(int)
    py = np.round(np.array([pr[1]/reduction for pr in data["prey_positions"][step_to_draw]])).astype(int)
    rrs, ccs = board.multi_circles(px, py, env_variables["prey_radius"])

    rrs = np.clip(rrs, 0, env_variables["arena_width"]-1)
    ccs = np.clip(ccs, 0, env_variables["arena_height"]-1)

    board.db[rrs, ccs] = (0, 0, 1)

    full_masked_image = board.get_masked_pixels(np.array([data["fish_position"][step_to_draw][0], data["fish_position"][step_to_draw][1]]),
                                                data["prey_positions"][step_to_draw],
                                                predator_bodies
                                                )

    test_eye_l.env_variables["show_noise"] = False
    test_eye_r.env_variables["show_noise"] = False
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

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(photons_l)
    axs[1].imshow(photons_r)
    axs[0].axes.get_yaxis().set_visible(False)
    axs[1].axes.get_yaxis().set_visible(False)
    axs[0].axes.get_xaxis().set_visible(False)
    axs[1].axes.get_xaxis().set_visible(False)
    plt.savefig("./Panels/Panel-1/observation_no_noise.jpg")
    plt.show()


# visualise_environent_at_step("dqn_scaffold_18-2", "Behavioural-Data-Free", "Naturalistic-3", 56)

visualise_environent_at_step("dqn_scaffold_18-2", "Behavioural-Data-Free", "Naturalistic-4", 40)




