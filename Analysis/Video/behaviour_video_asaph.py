import copy

import numpy as np
import math
import sys
import h5py
import json
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import matplotlib.pyplot as plt
import skimage.draw as draw
from skimage import io

from Analysis.load_data import load_data
from Configurations.Networks.original_network import base_network_layers, ops, connectivity
from skimage.transform import resize, rescale
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from scipy.signal import detrend
from scipy.stats import zscore



class DrawingBoard:

    def __init__(self, width, height, data, include_background):
        plt.style.use('dark_background')

        self.width = width
        self.height = height
        self.include_background = include_background
        if include_background:
            self.background = data["sediment"][:, :,]
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
        rr, cc = draw.ellipse(int(y_position), int(x_position), 3, 1, rotation=-fish_angle)
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
                 scale=1.0, draw_action_space_usage=True, trim_to_fish=True, save_id="", showed_region_quad=500, n_actions_to_show=20,
                 s_per_frame=0.03, include_background=True, as_gif=False, steps_to_show=None):
    #try:
    #    with open(f"../../Configurations/Assay-Configs/{config_name}_env.json", 'r') as f:
    #        env_variables = json.load(f)
    #except FileNotFoundError:
    #    with open(f"Configurations/Assay-Configs/{config_name}_env.json", 'r') as f:
    #        env_variables = json.load(f)


    rnn_states = data['rnn_state'][:, 0, :]
    rnn_states = detrend(zscore(rnn_states), axis=0)
    rnn_states = rnn_states[:, np.std(rnn_states, axis=0) > 1e-6]
    print(f"Used RNN dimensions: {rnn_states.shape[1]}")
    rnn_states_PCA = PCA(n_components=3).fit_transform(rnn_states)

    save_location += save_id
    if "Training-Output" in save_location:
        fast_mode = True
    else:
        fast_mode = False

    fig = plt.figure(facecolor='0.9', figsize=(4, 3))
    gs = fig.add_gridspec(nrows=10, ncols=10, left=0.05, right=0.85,
                      hspace=0.1, wspace=0.05)
    ax0 = fig.add_subplot(gs[:7, 0:6]) # arena
    
    ax1 = fig.add_subplot(gs[7, 0:3])   # left eye
    ax2 = fig.add_subplot(gs[7, 3:6])   # right eye
    ax3 = fig.add_subplot(gs[0, 6:8])  # energy
    ax3.set_facecolor((0,0,0))
    ax4 = fig.add_subplot(gs[1, 6:10])  # pca over time
    ax4.set_facecolor((0,0,0))
    ax5 = fig.add_subplot(gs[2:6, 6:10], projection='3d') # pca in 3d
    ax5.set_facecolor((0,0,0))

 #   ax5 = fig.add_subplot(gs[2, 6:10])
 #   ax6 = fig.add_subplot(gs[3, 6:10])
 #   ax7 = fig.add_subplot(gs[4, 6:10])
    ax8 = fig.add_subplot(gs[0, 8]) # action space usage
    if continuous_actions:
        ax9 = fig.add_subplot(gs[0, 9]) # action space usage (continuous)

    if not fast_mode:
        ax10 = fig.add_subplot(gs[6:10, 6:10]) # vision (polar)

    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata, codec='mpeg4', bitrate=1000000)

    board = DrawingBoard(env_variables["arena_width"], env_variables["arena_height"], data, include_background)
    if show_energy_state:
        energy_levels = data["energy_state"]
    fish_positions = data["fish_position"]
    num_steps = fish_positions.shape[0]
    if steps_to_show is None:
        steps_to_show = range(num_steps)
        
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
                
    #     frames = np.zeros((num_steps, np.int(env_variables["arena_height"]*scale), np.int((env_variables["arena_width"]+addon)*scale), 3))
    obs_len = data['observation'].shape[1]
    obs_atom = env_variables['visual_field'] / obs_len
    remaining = 360 - env_variables['visual_field']
    pie_sizes = np.ones(obs_len) * obs_atom
    pie_sizes = np.append(pie_sizes, remaining)


    with writer.saving(fig, f"{save_location}.mp4", 500):

        for step in steps_to_show:
            if "Training-Output" not in save_location:
                print(f"{step}/{num_steps}")
            if continuous_actions:
                action_buffer.append([data["impulse"][step], data["angle"][step]])
            else:
                action_buffer.append(data["action"][step])
                bins = np.arange(0, np.max(action_buffer)+1.5) - 0.5
                actions = np.arange(0, np.max(action_buffer)+1)

                action_hist = np.histogram(action_buffer, bins)[0]
                actions = actions[action_hist > 0]
                action_hist = action_hist[action_hist > 0]
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
                                                                                                  bkg_scatter=0.1,
                                                                                                  consumption_buffer=consumption_buffer)



            if show_energy_state:
                fish_body_colour = (1-energy_levels[step], energy_levels[step], 0)
            else:
                fish_body_colour = (0, 1, 0)

            board.fish_shape(fish_positions[step], env_variables['fish_mouth_radius'],
                                env_variables['fish_head_radius'], env_variables['fish_tail_length'],
                            (0, 1, 0), fish_body_colour, data["fish_angle"][step])

            # Draw prey
            px = np.round(np.array([pr[0] for pr in data["prey_positions"][step]])).astype(int)
            py = np.round(np.array([pr[1] for pr in data["prey_positions"][step]])).astype(int)
            rrs, ccs = board.multi_circles(px, py, env_variables["prey_radius_visualisation"])

            rrs = np.clip(rrs, 0, env_variables["arena_width"]-1)
            ccs = np.clip(ccs, 0, env_variables["arena_height"]-1)

            board.db[rrs, ccs] = (1, 0.5, 1)

            # Draw sand grains
            if env_variables["sand_grain_num"] > 0:
                px = np.round(np.array([pr[0] for pr in data["sand_grain_positions"][step]])).astype(int)
                py = np.round(np.array([pr[1] for pr in data["sand_grain_positions"][step]])).astype(int)
                rrs, ccs = board.multi_circles(px, py, env_variables["prey_radius_visualisation"])

                rrs = np.clip(rrs, 0, env_variables["arena_width"] - 1)
                ccs = np.clip(ccs, 0, env_variables["arena_height"] - 1)

                board.db[rrs, ccs] = (0, 1, 1)


            if data["predator_presence"][step]:
                board.circle(data["predator_positions"][step], env_variables['predator_radius'], (0, 1, 0))

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
                dist_x2 = env_variables["arena_width"] - centre_x
                dist_y1 = centre_y
                dist_y2 = env_variables["arena_height"] - centre_y
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
            ax0.imshow(np.clip(this_frame, 0, 1), interpolation='nearest', aspect='auto')
            ax0.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

            left_obs = data['observation_old'][step, :, :, 0].T

            right_obs = data['observation_old'][step, :, :, 1].T
            ax1.clear()
            ax2.clear()
            left_obs = np.clip(left_obs, 0, 255)
            right_obs = np.clip(right_obs, 0, 255)
            left_obs_imj = np.expand_dims(left_obs, axis=2)
            right_obs_imj = np.expand_dims(right_obs, axis=2)
            left_obs_imj = np.concatenate((left_obs_imj, left_obs_imj, left_obs_imj), axis=2)
            right_obs_imj = np.concatenate((right_obs_imj, right_obs_imj, right_obs_imj), axis=2)
            left_obs_imj[0,:,1:] = 0
            right_obs_imj[0,:,1:] = 0
            left_obs_imj[1,:,1] = 0
            right_obs_imj[1,:,1] = 0
            left_obs_imj[2,:,[0,2]] = 0
            right_obs_imj[2,:,[0,2]] = 0

            ax1.imshow(left_obs_imj, interpolation='nearest', aspect='auto', vmin=1, vmax=256)
            ax1.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax2.imshow(right_obs_imj, interpolation='nearest', aspect='auto', vmin=1, vmax=256)
            ax2.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

           

            if not fast_mode:
                ax10.clear()

                left_eye_pos = (+np.cos(np.pi / 2 - data["fish_angle"][step]) * env_variables['eyes_biasx']*0.05,
                                +np.sin(np.pi / 2 - data["fish_angle"][step]) * env_variables['eyes_biasx']*0.05)
                right_eye_pos = (-np.cos(np.pi / 2 - data["fish_angle"][step]) * env_variables['eyes_biasx']*0.05,
                                -np.sin(np.pi / 2 - data["fish_angle"][step]) * env_variables['eyes_biasx']*0.05)

                rad_div = 5
                left_eye_start_angle = +env_variables['visual_field']/2 + 90 - env_variables['eyes_verg_angle'] / 2 - np.rad2deg(data["fish_angle"][step])
                left_eye_start_angle = left_eye_start_angle % 360
                alphas = np.clip(left_obs, 0, 255) / 255
                colors = np.zeros(alphas.shape)
                colors = np.concatenate((colors, np.ones((3, 1))), axis=1)
                alphas = np.concatenate((alphas, np.zeros((3, 1))), axis=1)
                colors_red = np.array([1-colors[0,:], colors[0,:], colors[0,:], alphas[0,:]]).T
                colors_uv = np.array([1-colors[1,:], colors[1,:], 1-colors[1,:], alphas[1,:]]).T
                colors_red2 = np.array([colors[2,:], 1-colors[2,:], colors[2,:], alphas[2,:]]).T
                ax10.pie(pie_sizes, startangle=left_eye_start_angle, counterclock=False, radius=1.0/rad_div, colors=colors_red2, wedgeprops={"edgecolor": None, "width": 0.5/rad_div}, center = left_eye_pos)
                ax10.pie(pie_sizes, startangle=left_eye_start_angle, counterclock=False, radius=1.5/rad_div, colors=colors_uv, wedgeprops={"edgecolor": None, "width": 0.5/rad_div}, center = left_eye_pos)
                ax10.pie(pie_sizes, startangle=left_eye_start_angle, counterclock=False, radius=2.0/rad_div, colors=colors_red, wedgeprops={"edgecolor": None, "width": 0.5/rad_div}, center = left_eye_pos)
                ax10.scatter(left_eye_pos[0], left_eye_pos[1], s=10, c='w')

                right_eye_start_angle = +env_variables['visual_field']/2 - 90 + env_variables['eyes_verg_angle'] / 2 - np.rad2deg(data["fish_angle"][step])
                right_eye_start_angle = right_eye_start_angle % 360
                alphas = np.clip(right_obs, 0, 255) / 255
                colors = np.zeros(alphas.shape)
                colors = np.concatenate((colors, np.ones((3, 1))), axis=1)
                alphas = np.concatenate((alphas, np.zeros((3, 1))), axis=1)
                colors_red = np.array([1-colors[0,:], colors[0,:], colors[0,:], alphas[0,:]]).T
                colors_uv = np.array([1-colors[1,:], colors[1,:], 1-colors[1,:], alphas[1,:]]).T
                colors_red2 = np.array([colors[2,:], 1-colors[2,:], colors[2,:], alphas[2,:]]).T
                ax10.pie(pie_sizes, startangle=right_eye_start_angle, counterclock=False, radius=1.0/rad_div, colors=colors_red2, wedgeprops={"edgecolor": None, "width": 0.5/rad_div}, center = right_eye_pos)
                ax10.pie(pie_sizes, startangle=right_eye_start_angle, counterclock=False, radius=1.5/rad_div, colors=colors_uv, wedgeprops={"edgecolor": None, "width": 0.5/rad_div}, center = right_eye_pos)
                ax10.pie(pie_sizes, startangle=right_eye_start_angle, counterclock=False, radius=2.0/rad_div, colors=colors_red, wedgeprops={"edgecolor": None, "width": 0.5/rad_div}, center = right_eye_pos)
                ax10.scatter(right_eye_pos[0], right_eye_pos[1], s=10, c='w')

                ax10.set_xlim(-0.4, 0.4)
                ax10.set_ylim(-0.4, 0.4)

            plot_start = max(0, step - 100)
            ax3.clear()
            ax3.plot(energy_levels[plot_start:step], linewidth=0.5)
            ax3.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax4.clear()
            ax4.plot(rnn_states_PCA[plot_start:step, :], linewidth=0.5)
            ax4.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            ax5.clear()
            rgb_color = [1.0, 0.5, 0.5]
            line = Vanishing_Line(100, 50, rgb_color)
            line.set_data()
            line.set_data(rnn_states_PCA[:step,0], rnn_states_PCA[:step,1], rnn_states_PCA[:step,2])
            ax5.add_collection(line.get_LineCollection())
            ax5.scatter(rnn_states_PCA[step,0], rnn_states_PCA[step,1], rnn_states_PCA[step,2], color='red', s=2)
            ax5.set_xlim((np.min(rnn_states_PCA[:,0]), np.max(rnn_states_PCA[:,0])))
            ax5.set_ylim((np.min(rnn_states_PCA[:,1]), np.max(rnn_states_PCA[:,1])))
            ax5.set_zlim((np.min(rnn_states_PCA[:,2]), np.max(rnn_states_PCA[:,2])))
            ax5.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            
            # ax5.plot(data['rnn_state_actor'][plot_start:step, 0, 10:20], linewidth=0.5)
            # ax5.tick_params(left = False, right = False , labelleft = False ,
            #         labelbottom = False, bottom = False)
            # ax6.clear()
            # ax6.plot(data['rnn_state_actor'][plot_start:step, 0, 20:30], linewidth=0.5)
            # ax6.tick_params(left = False, right = False , labelleft = False ,
            #         labelbottom = False, bottom = False)
            # ax7.clear()
            # ax7.plot(data['rnn_state_actor'][plot_start:step, 0, 30:40], linewidth=0.5)
            # ax7.tick_params(left=False, right=False , labelleft=False, labelbottom=False, bottom=False)

            if step % 10 == 0:
                if continuous_actions:
                    ab = np.array(action_buffer)

                    ax8.clear()
                    ax8.hist(ab[:, 0], bins=20)
                    ax8.tick_params(axis='both', which='major', labelsize=3, length=1, width=0.1)
                    for _, spine in ax8.spines.items():
                        spine.set_linewidth(0.1)

                    ax9.clear()
                    ax9.hist(ab[:, 1], bins=20)
                    ax9.tick_params(axis='both', which='major', labelsize=3, length=1, width=0.1)
                    for _, spine in ax9.spines.items():
                        spine.set_linewidth(0.1)

                else:

                    ax8.clear()
                    ax8.pie(action_hist, labels=actions, textprops={'fontsize': 2})

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


class Vanishing_Line(object):
    def __init__(self, n_points, tail_length, rgb_color):
        self.n_points = int(n_points)
        self.tail_length = int(tail_length)
        self.rgb_color = rgb_color

    def set_data(self, x=None, y=None, z=None):
        if x is None or y is None or z is None:
            self.lc = Line3DCollection([])
        else:
            x = x[-self.n_points:]
            y = y[-self.n_points:]
            z = z[-self.n_points:]
            
            self.points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            
            self.segments = np.concatenate([self.points[:-1], self.points[1:]],
                                           axis=1)
            if hasattr(self, 'alphas'):
                del self.alphas
            if hasattr(self, 'rgba_colors'):
                del self.rgba_colors
            self.lc.set_segments(self.segments)
            self.lc.set_color(self.get_colors())
            self.lc.set_linewidth(0.5)

    def get_LineCollection(self):
        if not hasattr(self, 'lc'):
            self.set_data()
        return self.lc


    def get_alphas(self):
        n = len(self.points)
        if n < self.n_points:
            rest_length = self.n_points - self.tail_length
            if n <= rest_length:
                return np.ones(n)
            else:
                tail_length = n - rest_length
                tail = np.linspace(1./tail_length, 1., tail_length)
                rest = np.ones(rest_length)
                return np.concatenate((tail, rest))
        else: # n == self.n_points
            if not hasattr(self, 'alphas'):
                tail = np.linspace(1./self.tail_length, 1., self.tail_length)
                rest = np.ones(self.n_points - self.tail_length)
                self.alphas = np.concatenate((tail, rest))
            return self.alphas

    def get_colors(self):
        n = len(self.points)
        if  n < 2:
            return [self.rgb_color+[1.] for i in range(n)]
        if n < self.n_points:
            alphas = self.get_alphas()
            rgba_colors = np.zeros((n, 4))
            # first place the rgb color in the first three columns
            rgba_colors[:,0:3] = self.rgb_color
            # and the fourth column needs to be your alphas
            rgba_colors[:, 3] = alphas
            return rgba_colors
        else:
            if hasattr(self, 'rgba_colors'):
                pass
            else:
                alphas = self.get_alphas()
                rgba_colors = np.zeros((n, 4))
                # first place the rgb color in the first three columns
                rgba_colors[:,0:3] = self.rgb_color
                # and the fourth column needs to be your alphas
                rgba_colors[:, 3] = alphas
                self.rgba_colors = rgba_colors
            return self.rgba_colors



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
    data_file = sys.argv[1]
    config_file = sys.argv[2]

    if len(sys.argv) > 3:
        first_step = int(sys.argv[3])
        last_step = int(sys.argv[4])
        steps_to_show = range(first_step, last_step)
    else:
        steps_to_show = None
    #data_file = "../../Assay-Output/dqn_gamma-1/Behavioural-Data-Empty.h5"
    #config_file = f"../../Configurations/Assay-Configs/dqn_gamma_final_env.json"

    with open(config_file, 'r') as f:
        env_variables = json.load(f)
    with h5py.File(data_file, 'r') as datfl:
        group = list(datfl.keys())[0]
        data = {}
        for key in datfl[group].keys():
            data[key] = np.array(datfl[group][key])

    draw_episode(data, env_variables, 'test2', continuous_actions=False, show_energy_state=True,
                 trim_to_fish=True, showed_region_quad=500, include_background=True, steps_to_show=steps_to_show)



