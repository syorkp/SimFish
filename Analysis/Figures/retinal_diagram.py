import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_configuration_files
from Environment.Fish.eye import Eye

# from Analysis.Video.behaviour_video_construction import DrawingBoard
from Tools.drawing_board_new import NewDrawingBoard
from Analysis.load_data import load_data



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

# Build eyes
dark_col = int(env_variables['width'] * env_variables['dark_light_ratio'])
verg_angle = env_variables['eyes_verg_angle'] * (np.pi / 180)
retinal_field = env_variables['visual_field'] * (np.pi / 180)
test_eye_l = Eye(board, verg_angle, retinal_field, True, env_variables, dark_col, False)
test_eye_r = Eye(board, verg_angle, retinal_field, False, env_variables, dark_col, False)

l_angles = test_eye_l.uv_photoreceptor_angles
l_angles_red = test_eye_l.red_photoreceptor_angles

fig, ax = plt.subplots(figsize=(10, 3))
ax.eventplot(l_angles_red, colors=["r"], lineoffsets=[-1], linewidths=[6])
ax.eventplot(l_angles, lineoffsets=[0], linewidths=[6])
ax.eventplot(l_angles_red, colors=["orangered"], lineoffsets=[1], linewidths=[6])
plt.vlines(0, -2, 2, colors="black")
ax.axes.get_yaxis().set_visible(False)
plt.xlabel("Angle from Midline (radians)", fontsize=20)
plt.savefig("./Panels/Panel-1/left_retina.jpg")
plt.tight_layout()
plt.show()

l_angles = test_eye_r.uv_photoreceptor_angles
l_angles_red = test_eye_r.red_photoreceptor_angles

fig, ax = plt.subplots(figsize=(10, 3))
ax.eventplot(l_angles_red, colors=["r"], lineoffsets=[-1], linewidths=[6])
ax.eventplot(l_angles, lineoffsets=[0], linewidths=[6])
ax.eventplot(l_angles_red, colors=["orangered"], lineoffsets=[1], linewidths=[6])
plt.vlines(0, -2, 2, colors="black")
ax.axes.get_yaxis().set_visible(False)
plt.xlabel("Angle from Midline (radians)", fontsize=20)
plt.savefig("./Panels/Panel-1/right_retina.jpg")
plt.tight_layout()
plt.show()


