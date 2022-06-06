import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_configuration_files
from Environment.Fish.eye import Eye
from Tools.drawing_board_new import NewDrawingBoard


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


