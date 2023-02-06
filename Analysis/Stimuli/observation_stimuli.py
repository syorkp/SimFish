import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_assay_configuration_files

from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment


def get_prey_stimulus(x, y, sim_state):
    """Returns an observation for prey appearing at particular coordinates in visual field."""
    # Build the simulation

    x += 750
    y += 750

    sim_state.prey_bodies[0].position = [x, y]
    sim_state.board.erase(bkg=sim_state.env_variables['bkg_scatter'])
    sim_state.draw_shapes(visualisation=False)
    observation = sim_state.resolve_visual_input_new(save_frames=True, activations=[],
                                                                   internal_state=np.array([[0]]))

    return observation


def get_x_y_for_theta(theta):
    """Returns unit x-y for a given egocentric angle value."""
    normalised_theta = abs(theta)
    if normalised_theta > 90:
        normalised_theta -= 90

    x = np.cos(normalised_theta * np.pi / 180)
    y = np.sin(normalised_theta * np.pi / 180)

    if theta == -90:
        x = -1
        y = 0
        xy = [x, y]

    elif theta == 0:
        x = 0
        y = 1
        xy = [x, y]
    elif theta == 90:
        x = 1
        y = 0
        xy = [x, y]
    else:
        if theta < -90:
            x *= -1
            y *= -1
            xy = [x, y]

        elif theta < 0:
            y *= -1
            xy = [y, x]

        elif theta > 90:
            y *= -1
            xy = [x, y]

        else:
            xy = [y, x]

    return xy


def get_prey_stimuli_across_visual_field(prey_distance, intervals, model_name):
    """Returns intervals x observation for prey stimulus appearing across the visual field at a specified distance."""
    _, env, _, _, _ = load_assay_configuration_files(model_name)
    env["prey_num"] = 1
    sim_state = DiscreteNaturalisticEnvironment(env, realistic_bouts=True, draw_screen=False,
                                                using_gpu=False)
    sim_state.reset()

    sim_state.fish.body.position = [env["width"]/2, env["height"]/2]
    sim_state.fish.body.angle = 0

    orientation_values = np.linspace(-env["visual_field"]+env["eyes_verg_angle"]/2,
                                     env["visual_field"]-env["eyes_verg_angle"]/2, intervals)
    # orientation_values = np.linspace(-150, -90, intervals)
    x_y_vals = [get_x_y_for_theta(o) for o in orientation_values]
    x_y_vals = np.array(x_y_vals)
    x_y_vals *= prey_distance

    # plt.scatter(x_y_vals[:, 1], x_y_vals[:, 0])
    # plt.show()

    observations = np.array([get_prey_stimulus(x_y[1], x_y[0], sim_state) for x_y in x_y_vals])
    return observations


if __name__ == "__main__":
    observations = get_prey_stimuli_across_visual_field(50, 10, "dqn_scaffold_18-1")





