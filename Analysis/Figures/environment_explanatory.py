import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_configuration_files


# from Analysis.Video.behaviour_video_construction import DrawingBoard
from Tools.drawing_board_new import NewDrawingBoard
from Analysis.load_data import load_data



def visualise_environent_at_step(model_name, assay_config, assay_id, step_to_draw):
    data = load_data(model_name, assay_config, assay_id)
    learning_params, env_variables, n, b, c = load_configuration_files(model_name="dqn_scaffold_18-1")
    fish_body_colour = (0, 1, 0)
    board = NewDrawingBoard(env_variables["width"], env_variables["height"], env_variables["decay_rate"],
                                env_variables["uv_photoreceptor_rf_size"], False, False, 1,
                                light_gain=env_variables["light_gain"], visible_scatter=env_variables["bkg_scatter"],
                            light_gradient=env_variables["light_gradient"])

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


    # TODO: Make red walls thicker and more visible.
    # TODO: Also create image of zoomed in of fish
    # TODO: Print scatter, obstrution, background, and light gain masks in parallel.
    # TODO: Print version with PR occupancy also shown.
    # TODO: Get observation for the given step.
    # TODO: Get rid of ticks and instead show scale bar.

    plt.figure(figsize=(10, 10))
    plt.imshow(board.db_visualisation)
    plt.show()


visualise_environent_at_step("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1", 100)





