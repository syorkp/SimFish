import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_configuration_files
from Environment.Fish.eye import Eye
from Tools.drawing_board_new import NewDrawingBoard


def generate_wall_inputs_full_field(eye, drawing_board, width, height, env_variables):
    max_red_photons = np.zeros((int(width/20), int(height/20)))

    for w in range(3, (width-10)):
        for h in range(3, (height-10)):
            print(h)
            # TODO: Set eye orientation towards correct wall.
            fish_orientation = np.pi
            left_eye_pos = (
                +np.cos(np.pi / 2 - fish_orientation) * env_variables['eyes_biasx'] + w * 20,
                -np.sin(np.pi / 2 - fish_orientation) * env_variables['eyes_biasx'] + h * 20)

            masked_pixels = drawing_board.get_masked_pixels([w*20, h*20], np.array([]), np.array([]))
            eye.read(masked_pixels, left_eye_pos[0], left_eye_pos[1], fish_orientation)
            red_photons = eye.readings[:, 0]
            max_red_photons[w, h] = np.max(red_photons)

    plt.imshow(max_red_photons)
    plt.show()


def build_board_and_eye(env_variables):
    board = NewDrawingBoard(env_variables["width"], env_variables["height"], env_variables["decay_rate"],
                            env_variables["uv_photoreceptor_rf_size"], False, False, 1,
                            light_gain=env_variables["light_gain"], visible_scatter=env_variables["bkg_scatter"])

    verg_angle = 77. * (np.pi / 180)
    retinal_field = 163. * (np.pi / 180)
    dark_col = int(env_variables['width'] * env_variables['dark_light_ratio'])
    eye = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False)
    board.erase(env_variables["bkg_scatter"])
    return eye, board


learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_15-1")
eye, board = build_board_and_eye(env_variables)
generate_wall_inputs_full_field(eye, board, 1500, 1500, env_variables)


