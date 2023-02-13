import copy

import numpy as np
import math
import sys
import h5py
import json
import matplotlib.pyplot as plt
import skimage.draw as draw
from skimage import io

from Analysis.load_data import load_data
from Configurations.Networks.original_network import base_network_layers, ops, connectivity
from Tools.make_video import make_video
from skimage.transform import resize, rescale
from Tools.make_gif import make_gif
from matplotlib.animation import FFMpegWriter



class DrawingBoard:

    def __init__(self, width, height, data, include_background):
        self.width = width
        self.height = height
        self.include_background = include_background
        if include_background:
            self.background = data["background"][:, :,]
            self.background = np.expand_dims(self.background/10, 2)
            self.background = np.concatenate((self.background,
                                              self.background,
                                              np.zeros(self.background.shape)), axis=2)

        self.db = self.get_base_arena(0.3)


    def get_base_arena(self, bkg=0.3):
        db = (np.ones((self.height, self.width, 3), dtype=np.double) * bkg)
        db[1:2, :] = np.array([1, 0, 0])
        db[self.width - 2:self.width - 1, :] = np.array([1, 0, 0])
        db[:, 1:2] = np.array([1, 0, 0])
        db[:, self.height - 2:self.height - 1] = np.array([1, 0, 0])
        if self.include_background:
            db += self.background
        return db

    def circle(self, center, rad, color):
        rr, cc = draw.circle(center[1], center[0], rad, self.db.shape)
        self.db[rr, cc, :] = color

    @staticmethod
    def multi_circles(cx, cy, rad):
        rr, cc = draw.circle(0, 0, rad)
        rrs = np.tile(rr, (len(cy), 1)) + np.tile(np.reshape(cy, (len(cy), 1)), (1, len(rr)))
        ccs = np.tile(cc, (len(cx), 1)) + np.tile(np.reshape(cx, (len(cx), 1)), (1, len(cc)))
        return rrs, ccs

    def overlay_salt(self, salt_location):
        """Show salt source"""
        # Consider modifying so that shows distribution.
        self.circle(salt_location, 20, (0, 1, 1))

    def tail(self, head, left, right, tip, color):
        tail_coordinates = np.array((head, left, tip, right))
        rr, cc = draw.polygon(tail_coordinates[:, 1], tail_coordinates[:, 0], self.db.shape)
        self.db[rr, cc, :] = color

    def fish_shape(self, mouth_centre, mouth_rad, head_rad, tail_length, mouth_colour, body_colour, angle):
        offset = np.pi / 2
        angle += offset
        angle = -angle
        self.circle(mouth_centre, mouth_rad, mouth_colour)  # For the mouth.
        dx1, dy1 = head_rad * np.sin(angle), head_rad * np.cos(angle)
        head_centre = (mouth_centre[0] + dx1,
                       mouth_centre[1] + dy1)
        self.circle(head_centre, head_rad, body_colour)
        dx2, dy2 = -1 * dy1, dx1
        left_flank = (head_centre[0] + dx2,
                      head_centre[1] + dy2)
        right_flank = (head_centre[0] - dx2,
                       head_centre[1] - dy2)
        tip = (mouth_centre[0] + (tail_length + head_rad) * np.sin(angle),
               mouth_centre[1] + (tail_length + head_rad) * np.cos(angle))
        self.tail(head_centre, left_flank, right_flank, tip, body_colour)

    def show_action_continuous(self, impulse, angle, fish_angle, x_position, y_position, colour):
        # rr, cc = draw.ellipse(int(y_position), int(x_position), (abs(angle) * 3) + 3, (impulse*0.5) + 3, rotation=-fish_angle)
        rr, cc = draw.ellipse(int(y_position), int(x_position), 3, (impulse*0.5) + 3, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def show_action_discrete(self, fish_angle, x_position, y_position, colour):
        rr, cc = draw.ellipse(int(y_position), int(x_position), 5, 3, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def show_consumption(self, fish_angle, x_position, y_position, colour):
        rr, cc = draw.ellipse(int(y_position), int(x_position), 10, 6, rotation=-fish_angle)
        self.db[rr, cc, :] = colour

    def get_action_colour(self, action):
        """Returns the (R, G, B) for associated actions"""
        if action == 0:  # Slow2
            action_colour = (0, 1, 0)

        elif action == 1:  # RT right
            action_colour = (0, 1, 0)

        elif action == 2:  # RT left
            action_colour = (0, 1, 0)

        elif action == 3:  # Short capture swim
            action_colour = (1, 0, 1)

        elif action == 4:  # j turn right
            action_colour = (1, 1, 1)

        elif action == 5:  # j turn left
            action_colour = (1, 1, 1)

        elif action == 6:  # Do nothing
            action_colour = (0, 0, 0)

        elif action == 7:  # c start right
            action_colour = (1, 0, 0)

        elif action == 8:  # c start left
            action_colour = (1, 0, 0)

        elif action == 9:  # Approach swim.
            action_colour = (0, 1, 0)

        elif action == 10:  # j turn right (large)
            action_colour = (1, 1, 1)

        elif action == 11:  # j turn left (large)
            action_colour = (1, 1, 1)

        else:
            action_colour = (0, 0, 0)
            print("Invalid action given")

        return action_colour

    def apply_light(self, dark_col, dark_gain, light_gain, visualisation):
        if dark_col < 0:
            dark_col = 0
        if visualisation:
            if self.light_gradient > 0 and dark_col > 0:
                gradient = self.chosen_math_library.linspace(dark_gain, light_gain, self.light_gradient)
                gradient = self.chosen_math_library.expand_dims(gradient, 0)
                gradient = self.chosen_math_library.repeat(gradient, self.height, 0)
                gradient = self.chosen_math_library.expand_dims(gradient, 2)
                self.db_visualisation[:, int(dark_col-(self.light_gradient/2)):int(dark_col+(self.light_gradient/2))] *= gradient
                self.db_visualisation[:, :int(dark_col-(self.light_gradient/2))] *= dark_gain
                self.db_visualisation[:, int(dark_col+(self.light_gradient/2)):] *= light_gain
            else:
                self.db_visualisation[:, :dark_col] *= dark_gain
                self.db_visualisation[:, dark_col:] *= light_gain

        else:
            if self.light_gradient > 0 and dark_col > 0:
                gradient = self.chosen_math_library.linspace(dark_gain, light_gain, self.light_gradient)
                gradient = self.chosen_math_library.expand_dims(gradient, 0)
                gradient = self.chosen_math_library.repeat(gradient, self.height, 0)
                gradient = self.chosen_math_library.expand_dims(gradient, 2)
                self.db[:, int(dark_col-(self.light_gradient/2)):int(dark_col+(self.light_gradient/2))] *= gradient
                self.db[:, :int(dark_col-(self.light_gradient/2))] *= dark_gain
                self.db[:, int(dark_col+(self.light_gradient/2)):] *= light_gain
            else:
                self.db[:, :dark_col] *= dark_gain
                self.db[:, dark_col:] *= light_gain


def draw_previous_actions(board, past_actions, past_positions, fish_angles, adjusted_colour_index,
                          continuous_actions, n_actions_to_show, bkg_scatter, consumption_buffer=None):
    while len(past_actions) > n_actions_to_show:
        past_actions.pop(0)
    while len(past_positions) > n_actions_to_show:
        past_positions.pop(0)
    while len(fish_angles) > n_actions_to_show:
        fish_angles.pop(0)
    while len(consumption_buffer) > n_actions_to_show:
        consumption_buffer.pop(0)

    for i, a in enumerate(past_actions):
        if continuous_actions:
            if a[1] < 0:
                action_colour = (
                adjusted_colour_index, bkg_scatter, bkg_scatter)
            else:
                action_colour = (bkg_scatter, adjusted_colour_index, adjusted_colour_index)

            board.show_action_continuous(a[0], a[1], fish_angles[i], past_positions[i][0],
                                              past_positions[i][1], action_colour)
        else:
            action_colour = board.get_action_colour(past_actions[i])
            board.show_action_discrete(fish_angles[i], past_positions[i][0],
                                            past_positions[i][1], action_colour)
        if consumption_buffer is not None:
            if consumption_buffer[i] == 1:
                board.show_consumption(fish_angles[i], past_positions[i][0],
                                       past_positions[i][1], (1, 0, 0))
    return board, past_actions, past_positions, fish_angles


def draw_action_space_usage_continuous(current_height, current_width, action_buffer, max_impulse=10, max_angle=1):
    difference = 300
    extra_area = np.zeros((current_height, difference - 20, 3))
    available_height = current_height-20

    impulse_resolution = 4
    angle_resolution = 20

    # Create counts for binned actions
    impulse_bins = np.linspace(0, max_impulse, int(max_impulse * impulse_resolution))
    binned_impulses = np.digitize(np.array(action_buffer)[:, 0], impulse_bins)
    impulse_bin_counts = np.array([np.count_nonzero(binned_impulses == i) for i in range(len(impulse_bins))]).astype(float)

    angle_bins = np.linspace(-max_angle, max_angle, int(max_angle * angle_resolution))
    binned_angles = np.digitize(np.array(action_buffer)[:, 1], angle_bins)
    angle_bin_counts = np.array([np.count_nonzero(binned_angles == i) for i in range(len(angle_bins))]).astype(float)

    impulse_bin_scaling = (difference-20)/max(impulse_bin_counts)
    angle_bin_scaling = (difference-20)/max(angle_bin_counts)

    impulse_bin_counts *= impulse_bin_scaling
    angle_bin_counts *= angle_bin_scaling

    impulse_bin_counts = np.floor(impulse_bin_counts).astype(int)
    angle_bin_counts = np.floor(angle_bin_counts).astype(int)

    bin_height = int(math.floor(available_height / (len(impulse_bin_counts) + len(angle_bin_counts))))

    current_h = 0
    for count in impulse_bin_counts:
        extra_area[current_h:current_h+bin_height, 0:count, :] = 255.0
        current_h += bin_height

    current_h += 100

    for count in angle_bin_counts:
        extra_area[current_h:current_h+bin_height, 0:count, :] = 255.0
        current_h += bin_height

    x = extra_area[:, :, 0]

    return extra_area


def draw_action_space_usage_discrete(current_height, current_width, action_buffer):
    difference = 300
    extra_area = np.zeros((current_height, difference - 20, 3))

    action_bins = [i for i in range(10)]
    action_bin_counts = np.array([np.count_nonzero(np.array(action_buffer) == i) for i in action_bins]).astype(float)

    action_bin_scaling = (difference-20)/max(action_bin_counts)
    action_bin_counts *= action_bin_scaling
    action_bin_counts = np.floor(action_bin_counts).astype(int)

    bin_height = int(math.floor(current_width/len(action_bins)))

    current_h = 0

    for count in action_bin_counts:
        extra_area[current_h:current_h+bin_height, 0:count, :] = 255.0
        current_h += bin_height

    return extra_area


def draw_episode(data, env_variables, save_location, continuous_actions, draw_past_actions=True, show_energy_state=True,
                 scale=1.0, draw_action_space_usage=True, trim_to_fish=True, save_id="", showed_region_quad=500, n_actions_to_show=500,
                 s_per_frame=0.03, include_background=True, as_gif=False):
    #try:
    #    with open(f"../../Configurations/Assay-Configs/{config_name}_env.json", 'r') as f:
    #        env_variables = json.load(f)
    #except FileNotFoundError:
    #    with open(f"Configurations/Assay-Configs/{config_name}_env.json", 'r') as f:
    #        env_variables = json.load(f)
    save_location += save_id

    fig = plt.figure(facecolor='0.9', figsize=(4, 3))
    gs = fig.add_gridspec(nrows=9, ncols=9, left=0.05, right=0.85,
                      hspace=0.1, wspace=0.05)
    ax0 = fig.add_subplot(gs[:7, 0:6])
    
    ax1 = fig.add_subplot(gs[7, 0:4])
    ax2 = fig.add_subplot(gs[7, 4:8])
    ax11 = fig.add_subplot(gs[8, 0:4])
    ax22 = fig.add_subplot(gs[8, 4:8])
    ax3 = fig.add_subplot(gs[0, 6:])
    ax4 = fig.add_subplot(gs[1, 6:])
    ax5 = fig.add_subplot(gs[2, 6:])
    ax6 = fig.add_subplot(gs[3, 6:])
    ax7 = fig.add_subplot(gs[4, 6:])
    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata, codec='mpeg4')

    board = DrawingBoard(env_variables["width"], env_variables["height"], data, include_background)
    if show_energy_state:
        energy_levels = data["energy_state"]
    fish_positions = data["fish_position"]
    num_steps = fish_positions.shape[0]

    #frames = []
    action_buffer = []
    position_buffer = []
    orientation_buffer = []
    consumption_buffer = []
    # if trim_to_fish:
    #     frames = np.zeros((num_steps, np.int(scale * showed_region_quad*2), np.int(scale * showed_region_quad*2), 3))
    # else:
    #     if draw_action_space_usage:
    #         addon = 300
    #     else:
    #         addon = 0
                
    #     frames = np.zeros((num_steps, np.int(env_variables["height"]*scale), np.int((env_variables["width"]+addon)*scale), 3))
    with writer.saving(fig, f"{save_location}.mp4", 300):

        for step in range(num_steps):
            if "Training-Output" not in save_location:
                print(f"{step}/{num_steps}")
            if continuous_actions:
                action_buffer.append([data["impulse"][step], data["angle"][step]])
            else:
                action_buffer.append(data["action"][step])
            position_buffer.append(fish_positions[step])
            orientation_buffer.append(data["fish_angle"][step])
            consumption_buffer.append(data["consumed"][step])

            if draw_past_actions:
                # adjusted_colour_index = ((1 - env_variables["bkg_scatter"]) * (step + 1) / n_actions_to_show) + \
                #                         env_variables["bkg_scatter"]
                # adjusted_colour_index = (1 - env_variables["bkg_scatter"]) + env_variables["bkg_scatter"]
                adjusted_colour_index = 1
                board, action_buffer, position_buffer, orientation_buffer = draw_previous_actions(board, action_buffer,
                                                                                                  position_buffer, orientation_buffer,
                                                                                                  adjusted_colour_index=adjusted_colour_index,
                                                                                                  continuous_actions=continuous_actions,
                                                                                                  n_actions_to_show=n_actions_to_show,
                                                                                                  bkg_scatter=env_variables["bkg_scatter"],
                                                                                                  consumption_buffer=consumption_buffer)



            if show_energy_state:
                fish_body_colour = (1-energy_levels[step], energy_levels[step], 0)
            else:
                fish_body_colour = (0, 1, 0)

            board.fish_shape(fish_positions[step], env_variables['fish_mouth_size'],
                                env_variables['fish_head_size'], env_variables['fish_tail_length'],
                            (0, 1, 0), fish_body_colour, data["fish_angle"][step])

            # Draw prey
            px = np.round(np.array([pr[0] for pr in data["prey_positions"][step]])).astype(int)
            py = np.round(np.array([pr[1] for pr in data["prey_positions"][step]])).astype(int)
            rrs, ccs = board.multi_circles(px, py, env_variables["prey_size_visualisation"])

            rrs = np.clip(rrs, 0, env_variables["width"]-1)
            ccs = np.clip(ccs, 0, env_variables["height"]-1)

            board.db[rrs, ccs] = (0, 0, 1)

            # Draw sand grains
            if env_variables["sand_grain_num"] > 0:
                px = np.round(np.array([pr.position[0] for pr in data["sand_grain_positions"]])).astype(int)
                py = np.round(np.array([pr.position[1] for pr in data["sand_grain_positions"]])).astype(int)
                rrs, ccs = board.multi_circles(px, py, env_variables["prey_size_visualisation"])

                rrs = np.clip(rrs, 0, env_variables["width"] - 1)
                ccs = np.clip(ccs, 0, env_variables["height"] - 1)

                board.db_visualisation[rrs, ccs] = (0, 0, 1)


            if data["predator_presence"][step]:
                board.circle(data["predator_positions"][step], env_variables['predator_size'], (0, 1, 0))

            # if draw_action_space_usage:
            #     if continuous_actions:
            #         action_space_strip = draw_action_space_usage_continuous(board.db.shape[0], board.db.shape[1], action_buffer)
            #     else:
            #         action_space_strip = draw_action_space_usage_discrete(board.db.shape[0], board.db.shape[1], action_buffer)

            #     frame = np.hstack((board.db, np.zeros((board.db.shape[0], 20, 3)), action_space_strip))
            # else:
            frame = board.db

            if trim_to_fish:
                centre_y, centre_x = fish_positions[step][0], fish_positions[step][1]
                # print(centre_x, centre_y)
                dist_x1 = centre_x
                dist_x2 = env_variables["width"] - centre_x
                dist_y1 = centre_y
                dist_y2 = env_variables["height"] - centre_y
                # print(dist_x1, dist_x2, dist_y1, dist_y2)
                if dist_x1 < showed_region_quad:
                    centre_x += showed_region_quad - dist_x1
                elif dist_x2 < showed_region_quad:
                    centre_x -= showed_region_quad - dist_x2
                if dist_y1 < showed_region_quad:
                    centre_y += showed_region_quad - dist_y1
                if dist_y2 < showed_region_quad:
                    centre_y -= showed_region_quad - dist_y2
                centre_x = int(centre_x)
                centre_y = int(centre_y)
                # Compute centre position - so can deal with edges
                frame = frame[centre_x-showed_region_quad:centre_x+showed_region_quad,
                        centre_y-showed_region_quad:centre_y+showed_region_quad]

            this_frame = rescale(frame, scale, multichannel=True, anti_aliasing=True)
            ax0.clear()
            ax0.imshow(this_frame, interpolation='nearest', aspect='auto')
            ax0.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

            left_obs = data['observation'][step, :, :, 0].T

            right_obs = data['observation'][step, :, :, 1].T
            ax1.clear()
            ax2.clear()
            ax1.imshow(left_obs, interpolation='nearest', aspect='auto', vmin=1, vmax=256)
            ax1.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax2.imshow(right_obs, interpolation='nearest', aspect='auto', vmin=1, vmax=256)
            ax2.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

            left_obs_c = data['observation_classic'][step, :, :, 0].T

            right_obs_c = data['observation_classic'][step, :, :, 1].T
            ax11.clear()
            ax22.clear()
            ax11.imshow(left_obs_c, interpolation='nearest', aspect='auto', vmin=1, vmax=256)
            ax11.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax22.imshow(right_obs_c, interpolation='nearest', aspect='auto', vmin=1, vmax=256)
            ax22.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

            plot_start = max(0, step - 100)
            ax3.clear()
            ax3.plot(energy_levels[plot_start:step], color='green')
            ax3.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax4.clear()
            ax4.plot(data['rnn_state_actor'][plot_start:step, 0, :10])
            ax4.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax5.clear()
            ax5.plot(data['rnn_state_actor'][plot_start:step, 0, 10:20])
            ax5.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax6.clear()
            ax6.plot(data['rnn_state_actor'][plot_start:step, 0, 20:30])
            ax6.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax7.clear()
            ax7.plot(data['rnn_state_actor'][plot_start:step, 0, 30:40])
            ax7.tick_params(left=False, right=False , labelleft=False, labelbottom=False, bottom=False)
            #plt.draw()
            #plt.pause(0.001)
            board.db = board.get_base_arena(0.3)
            writer.grab_frame()

            board.db = board.get_base_arena(0.3)

    #frames *= 255

    #if training_episode:
    #    save_location = f"Training-Output/{model_name}/episodes/{save_id}"
    #else:
    #    save_location = f"Analysis-Output/Behavioural/Videos/{model_name}-{save_id}-behaviour"

    # if as_gif:
    #     make_gif(frames, f"{save_location}.gif", duration=len(frames) * s_per_frame, true_image=True)
    # else:
    #     make_video(frames, f"{save_location}.mp4", duration=len(frames) * s_per_frame, true_image=True)


if __name__ == "__main__":
    # model_name = "dqn_scaffold_26-2"
    # data = load_data(model_name, "Behavioural-Data-Videos-A1", "Naturalistic-1")
    # assay_config_name = "dqn_26_2_videos"
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750)
    # model_name = "ppo_scaffold_21-2"
    # data = load_data(model_name, "Behavioural-Data-Videos-A1", "Naturalistic-5")
    # assay_config_name = "ppo_21_2_videos"
    # draw_episode(data, assay_config_name, model_name, continuous_actions=True, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="A15")
    # model_name = "dqn_scaffold_14-1"
    # data = load_data(model_name, "Interruptions-HA", "Naturalistic-3")
    # assay_config_name = "dqn_14_1"
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="Interrupted-3")

    # model_name = "dqn_scaffold_14-1"
    # data = load_data(model_name, "Interruptions-HA", "Naturalistic-5")
    # assay_config_name = "dqn_14_1"
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="Interrupted-5")
    # data_file = sys.argv[1]
    # config_file = sys.argv[2]

    data_file = "../../Assay-Output/dqn_gamma-1/Behavioural-Data-Empty.h5"
    config_file = f"../../Configurations/Assay-Configs/dqn_gamma_final_env.json"

    with open(config_file, 'r') as f:
        env_variables = json.load(f)
    with h5py.File(data_file, 'r') as datfl:
        group = list(datfl.keys())[0]
        data = {}
        for key in datfl[group].keys():
            data[key] = np.array(datfl[group][key])

    draw_episode(data, env_variables, '', continuous_actions=False, show_energy_state=True,
                 trim_to_fish=True, showed_region_quad=500, include_background=True)




