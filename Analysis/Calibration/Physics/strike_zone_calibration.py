import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from Environment.Action_Space.plot_bout_data import get_bout_data
from Analysis.Calibration.Physics.test_environment import TestEnvironment


def plot_strike_zone(fraction_capture_permitted, angle_deviation_allowed, n_repeats=1, use_action_means=True,
                     continuous=False, overlay_all_sCS_data=False, set_impulse=2., set_angle=0., impulse_effect_noise=0.,
                     angular_effect_noise=0.):
    env = TestEnvironment(fraction_capture_permitted, angle_deviation_allowed)

    xp, yp = np.arange(-5, 30, 0.5), np.arange(-8, 8, 0.5)
    resolution = 20

    xpe, ype = np.meshgrid(xp, yp)
    vectors1 = np.concatenate((np.expand_dims(xpe, 2), np.expand_dims(ype, 2)), axis=2)
    vectors = np.reshape(vectors1, (-1, 2))
    successful_capture_count = np.zeros((vectors.shape[0]))
    angs = np.zeros((vectors.shape[0]))
    n_test = vectors.shape[0]

    fish_positions = []

    for j in range(n_repeats):
        print(f"{j+1} / {n_repeats}")
        for i, vector in enumerate(vectors):
            # print(f"{i} / {n_test}")
            # Apply motor effect noise
            if impulse_effect_noise > 0:
                impulse = set_impulse + (np.random.normal(0, impulse_effect_noise) * abs(set_impulse))
                # Effect on impulse from angle
                impulse = impulse + (np.random.normal(0, 0.5) * abs(set_angle))
            else:
                impulse = set_impulse
            if angular_effect_noise > 0:
                angle = set_angle + (np.random.normal(0, angular_effect_noise) * abs(set_angle))
                # Effect on impulse from angle
                angle = angle + (np.random.normal(0, 0.02) * abs(set_impulse))
            else:
                angle = set_angle

            s = env.run_prey_capture(vector, fixed_capture=use_action_means, continuous=continuous, set_impulse=impulse,
                        set_angle=angle)

            if s:
                # fish_positions.append(np.array(env.body.position))
                successful_capture_count[i] += 1
                angs[i] = env.latest_incidence

    fish_positions = np.array(fish_positions)
    successful_capture_count = np.reshape(successful_capture_count, (vectors1.shape[0], vectors1.shape[1]))
    angs = np.reshape(angs, (vectors1.shape[0], vectors1.shape[1]))
    successful_capture_count /= n_repeats

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(successful_capture_count)

    # Display fish.
    x, y = 10, 16
    mouth_size, head_size, tail_length = env.env_variables['fish_mouth_radius'], env.env_variables['fish_head_radius'], \
                                         env.env_variables['fish_tail_length']
    mouth_size *= resolution/10
    head_size *= resolution/10
    tail_length *= resolution/10

    mouth_centre = (x, y)
    mouth = plt.Circle(mouth_centre, mouth_size, fc="green")
    ax.add_patch(mouth)

    angle = (1.5 * np.pi)
    dx1, dy1 = head_size * np.sin(angle), head_size * np.cos(angle)
    head_centre = (mouth_centre[0] + dx1,
                   mouth_centre[1] + dy1)
    head = plt.Circle(head_centre, head_size, fc="green")
    ax.add_patch(head)

    dx2, dy2 = -1 * dy1, dx1
    left_flank = (head_centre[0] + dx2,
                  head_centre[1] + dy2)
    right_flank = (head_centre[0] - dx2,
                   head_centre[1] - dy2)
    tip = (mouth_centre[0] + (tail_length + head_size) * np.sin(angle),
           mouth_centre[1] + (tail_length + head_size) * np.cos(angle))
    tail = plt.Polygon(np.array([left_flank, right_flank, tip]), fc="green")
    ax.add_patch(tail)

    scale_bar = AnchoredSizeBar(ax.transData,
                                resolution, '1mm', 'lower right',
                                pad=1,
                                color='red',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 16}
                                )
    ax.add_artist(scale_bar)

    # fish_positions -= 500
    # fish_positions[:, 0] = np.absolute(fish_positions[:, 0])
    # fish_positions *= resolution/10
    # fish_positions[:, 0] += x
    # fish_positions[:, 1] += y

    # plt.scatter(fish_positions[:, 0], fish_positions[:, 1], alpha=0.3)

    if overlay_all_sCS_data:
        distances, angles = get_bout_data(3)
        x_diff = distances * np.sin(angles)
        y_diff = distances * np.cos(angles)

        x_loc = x + (y_diff * resolution)
        y_loc = y + (x_diff * resolution)

        # Mirror yloc
        y_loc2 = y - (x_diff * resolution)
        x_loc = np.concatenate((x_loc, x_loc))
        y_loc = np.concatenate((y_loc, y_loc2))
        density_list = np.concatenate((np.expand_dims(x_loc, 1), np.expand_dims(y_loc, 1)), axis=1)

        plt.scatter(x_loc, y_loc, alpha=0.3)

        # Show density
        x = np.array([i[0] for i in density_list])
        y = np.array([i[1] for i in density_list])
        y = np.negative(y)
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        nbins = 300
        k = kde.gaussian_kde([y, x])
        yi, xi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="Reds", )  # cmap='gist_gray')#  cmap='PuBu_r')
        # plt.contour(yi, -xi, zi.reshape(xi.shape), 3)
        #
        # plt.xlim(0, successful_capture_count.shape[1]-1)
        # plt.ylim(0, successful_capture_count.shape[0]-1)


    fig.colorbar(im, ax=ax, label="Prop Success")

    plt.show()

    return angs
    # Run for many different drawn parameters, plotting scatter of whether was successful capture.
    # Run the same but in the reverse - not var


if __name__ == "__main__":
    # For PPO
    # Best Strike zone fit
    # angs = plot_strike_zone(fraction_capture_permitted=0.2, angle_deviation_allowed=np.pi / 5, n_repeats=10,
    #                         use_action_means=False, continuous=True, overlay_all_sCS_data=False, set_impulse=3.295740136878764,
    #                         set_angle=0.,
    #                         impulse_effect_noise=0.1,
    #                         angular_effect_noise=0.6)
    # #
    # # # For Allowing low angle captures
    # angs = plot_strike_zone(fraction_capture_permitted=0.2, angle_deviation_allowed=np.pi / 5, n_repeats=10,
    #                         use_action_means=False, continuous=True, overlay_all_sCS_data=False, set_impulse=3.295740136878764,
    #                         set_angle=0.1,
    #                         impulse_effect_noise=0.1,
    #                         angular_effect_noise=0.6)
    # #
    # # # For Forbidding high angle captures
    # angs = plot_strike_zone(fraction_capture_permitted=0.2, angle_deviation_allowed=np.pi / 5, n_repeats=10,
    #                         use_action_means=False, continuous=True, overlay_all_sCS_data=False, set_impulse=3.295740136878764,
    #                         set_angle=0.4,
    #                         impulse_effect_noise=0.1,
    #                         angular_effect_noise=0.6)
    #
    # # For Forbidding high impulse captures
    # angs = plot_strike_zone(fraction_capture_permitted=0.2, angle_deviation_allowed=np.pi / 5, n_repeats=10,
    #                         use_action_means=False, continuous=True, overlay_all_sCS_data=False, set_impulse=10,
    #                         set_angle=0.0,
    #                         impulse_effect_noise=0.1,
    #                         angular_effect_noise=0.6)

    # For DQN
    plot_strike_zone(fraction_capture_permitted=0.2, angle_deviation_allowed=np.pi/5,
                     n_repeats=10,
                     use_action_means=False, continuous=False, overlay_all_sCS_data=False)

