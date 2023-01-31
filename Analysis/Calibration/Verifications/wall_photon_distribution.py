import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_assay_configuration_files
from Environment.Fish.eye import Eye
from Tools.drawing_board_new import DrawingBoard


def get_orientation_to_closest_wall(width, height, x, y):
    if x < y:
        lower = True
    else:
        lower = False
    if x + y < width:
        closer = True
    else:
        closer = False

    if lower and closer:
        return np.pi
    elif not lower and closer:
        return np.pi * 1.5
    elif lower and not closer:
        return np.pi * 0.5
    else:
        return 0


def generate_wall_inputs_full_field(eye, drawing_board, width, height, env_variables, scaling=100):
    max_red_photons = np.zeros((int(width/scaling), int(height/scaling)))
    total = (width/scaling) * (height/scaling)
    total_remaining = (width/scaling) * (height/scaling)

    for w in range(1, int((width-3)/scaling)):
        for h in range(1, int((height-3)/scaling)):
            total_remaining -= 1
            fish_orientation = get_orientation_to_closest_wall(width, height, w*scaling, h*scaling)
            print(f"{(total_remaining/total) * 100}% remaining, Ori: {fish_orientation}")
            left_eye_pos = (
                +np.cos(np.pi / 2 - fish_orientation) * env_variables['eyes_biasx'] + w * scaling,
                -np.sin(np.pi / 2 - fish_orientation) * env_variables['eyes_biasx'] + h * scaling)

            masked_pixels = drawing_board.get_masked_pixels([w*scaling, h*scaling], np.array([]), np.array([]))
            eye.read(masked_pixels, left_eye_pos[0], left_eye_pos[1], fish_orientation)
            red_photons = eye.readings[:, 0]
            max_red_photons[w, h] = np.max(red_photons)

    plt.imshow(max_red_photons)
    plt.show()

    plt.plot(max_red_photons[:, int(width/(scaling*2))])
    plt.show()


def build_board_and_eye(env_variables):
    board = DrawingBoard(env_variables["width"], env_variables["height"], env_variables["decay_rate"],
                            env_variables["uv_photoreceptor_rf_size"], False, False, 1,
                            light_gain=env_variables["light_gain"], visible_scatter=env_variables["bkg_scatter"])

    verg_angle = 77. * (np.pi / 180)
    retinal_field = 163. * (np.pi / 180)
    dark_col = int(env_variables['width'] * env_variables['dark_light_ratio'])
    eye = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False)
    board.erase(env_variables["bkg_scatter"])
    return eye, board


learning_params, env_variables, n, b, c = load_assay_configuration_files("dqn_scaffold_15-1")
env_variables["bkg_scatter"] = 0
eye, board = build_board_and_eye(env_variables)
generate_wall_inputs_full_field(eye, board, 1500, 1500, env_variables)


