import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_configuration_files
from Environment.Fish.eye import Eye
from Tools.drawing_board_new import NewDrawingBoard


def generate_wall_inputs_full_field(eye, drawing_board, width, height, env_variables):
    red_photons = np.zeros((width, height))

    for w in range(width):
        for h in range(height):
            observation = eye.read()

def build_board_and_eye(env_variables):
    board = NewDrawingBoard(width, height, decay_rate, pr_size, False, False, 1, light_gain=luminance, visible_scatter=bkg_scatter)

    verg_angle = 77. * (np.pi / 180)
    retinal_field = 163. * (np.pi / 180)
    dark_col = int(env_variables['width'] * env_variables['dark_light_ratio'])
    eye = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False)

    return eye



