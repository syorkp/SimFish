
# To store from V1 no longer used parameters


env = {
    'num_photoreceptors': 120,  # number of visual 'rays' per eye
    'min_vis_dist': 20,
    'max_vis_dist': 180,

    'prey_impulse_rate': 0.25,  # fraction of prey receiving impulse per step

    'forward_swim_cost': 3,
    'forward_swim_impulse': 10,
    'routine_turn_cost': 3,
    'routine_turn_impulse': 5,
    'routine_turn_dir_change': 0.6,
    'capture_swim_cost': 5,
    'capture_swim_impulse': 5,
    'j_turn_cost': 2.5,
    'j_turn_impulse': 0.1,
    'j_turn_dir_change': 0.07,
    'rest_cost': 2,

    'distance_penalty_scaling_factor': 1.0,
    # NOTE: THESE ARE IGNORED IN NEW SIMULATION, where penalties are set by energy system.
    'angle_penalty_scaling_factor': 0.5,
    # NOTE: THESE ARE IGNORED IN NEW SIMULATION, where penalties are set by energy system.

}


# To store from V2 tried and no longer supported.



