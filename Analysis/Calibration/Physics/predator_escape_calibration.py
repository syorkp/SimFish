import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from Analysis.Behavioural.VisTools.get_action_name import get_action_name_unlateralised
from Analysis.Calibration.Physics.test_environment import TestEnvironment



def plot_escape_success(n_repeats=1, use_action_means=True,
                        continuous=False, set_impulse=2., set_angle=0., set_action=True, impulse_effect_noise=0.,
                        angular_effect_noise=0., predator_impulse=2., specified_action=7):
    env = TestEnvironment(predator_impulse)
    xp, yp = np.arange(-100, 100, 10), np.arange(-100, 100, 10)
    resolution = 1

    xpe, ype = np.meshgrid(xp, yp)
    vectors1 = np.concatenate((np.expand_dims(xpe, 2), np.expand_dims(ype, 2)), axis=2)
    vectors = np.reshape(vectors1, (-1, 2))
    successful_escape_count = np.zeros((vectors.shape[0]))
    pred_final_positions = []

    for j in range(n_repeats):
        print(f"{j} / {n_repeats}")
        for i, vector in enumerate(vectors):
            # print(f"{i} / {n_test}")
            # Apply motor effect noise
            if continuous:
                if impulse_effect_noise > 0:
                    impulse = set_impulse + (np.random.normal(0, impulse_effect_noise) * abs(set_impulse))
                    impulse = impulse + (np.random.normal(0, 0.5) * abs(set_angle))

                else:
                    impulse = set_impulse
                if angular_effect_noise > 0:
                    angle = set_angle + (np.random.normal(0, angular_effect_noise) * abs(set_angle))
                    angle = angle + (np.random.normal(0, 0.02) * abs(set_impulse))

                else:
                    angle = set_angle
            else:
                if set_action:
                    impulse = set_impulse
                    angle = set_angle
                else:
                    impulse = None
                    angle = None

            s, fish_pos, pred_pos = env.run_predator_escape(vector, fixed_action=use_action_means, continuous=continuous, set_impulse=impulse,
                        set_angle=angle, specified_action=specified_action)
            pred_final_positions.append(pred_pos)

            if s:
                successful_escape_count[i] += 1

    successful_escape_count = np.reshape(successful_escape_count, (vectors1.shape[0], vectors1.shape[1]))
    successful_escape_count /= n_repeats

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(successful_escape_count)

    # Display fish.
    x, y = 10, 10
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

    # Predator
    predator_size = 32 * resolution/10
    dx1, dy1 = head_size * np.sin(angle), head_size * np.cos(angle)
    predator_centre = (5 + dx1,
                       5 + dy1)
    predator = plt.Circle(predator_centre, predator_size/2, fc="red")
    ax.add_patch(predator)

    scale_bar = AnchoredSizeBar(ax.transData,
                                resolution, '1mm', 'lower right',
                                pad=1,
                                color='red',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 16}
                                )
    ax.add_artist(scale_bar)
    fig.colorbar(im, ax=ax, label="Prop Success")
    fish_pos /= 10/resolution
    fish_pos -= 50
    fish_pos += 10
    #
    # pred_final_positions /= 10/resolution
    # pred_final_positions -= 50
    # pred_final_positions += 40

    plt.scatter(fish_pos[:, 0], fish_pos[:, 1])
    plt.title(f"Bout: {get_action_name_unlateralised(specified_action)} ")
    # plt.scatter(pred_final_positions[:, 0], pred_final_positions[:, 1], alpha=0.1)
    plt.show()


if __name__ == "__main__":
    #           PPO
    # Simulating rest
    # plot_escape_success(n_repeats=10,
    #                     use_action_means=False, continuous=True, set_impulse=0,
    #                     set_angle=0,
    #                     impulse_effect_noise=0.1, angular_effect_noise=0.6, predator_impulse=25)
    # Simulating SLC
    # plot_escape_success(n_repeats=10,
    #                     use_action_means=False, continuous=True, set_impulse=7.03322223 * 3.4452532909386484,
    #                     set_angle=0.67517832,
    #                     impulse_effect_noise=0.1, angular_effect_noise=0.6, predator_impulse=25)
    # Simulating RT
    # plot_escape_success(n_repeats=10,
    #                     use_action_means=False, continuous=True, set_impulse=2.74619216 * 3.4452532909386484,
    #                     set_angle=0.82713249,
    #                     impulse_effect_noise=0.1, angular_effect_noise=0.6, predator_impulse=25)
    # Simulating Scs
    # plot_escape_success(n_repeats=10,
    #                     use_action_means=False, continuous=True, set_impulse=2.49320953e+00 * 3.4452532909386484,
    #                     set_angle=2.36217665e-19,
    #                     impulse_effect_noise=0.1, angular_effect_noise=0.6, predator_impulse=25)

    #           DQN
    plot_escape_success(n_repeats=10,
                        use_action_means=False, continuous=False, set_impulse=0,
                        set_angle=0.0,
                        impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=6., specified_action=1)
    # plot_escape_success(n_repeats=10,
    #                     use_action_means=False, continuous=False, set_impulse=0,
    #                     set_angle=0.0,
    #                     impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=25., specified_action=6)
    # plot_escape_success(n_repeats=10,
    #                     use_action_means=False, continuous=False, set_impulse=0,
    #                     set_angle=0.0,
    #                     impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=25., specified_action=0)
    plot_escape_success(n_repeats=10,
                        use_action_means=False, continuous=False, set_impulse=0,
                        set_angle=0.0,
                        impulse_effect_noise=0.14, angular_effect_noise=0.5, predator_impulse=6., specified_action=7)

