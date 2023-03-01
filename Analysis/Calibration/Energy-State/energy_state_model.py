import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_assay_configuration_files
from Environment.Fish.fish import Fish
from Environment.Fish.continuous_fish import ContinuousFish
from Tools.drawing_board import DrawingBoard


def build_fish_model(env_variables, continuous):

    db = DrawingBoard(env_variables["width"],
                         env_variables["height"],
                         env_variables["light_decay_rate"],
                         env_variables["uv_photoreceptor_rf_size"],
                         False,
                         False)
    if continuous:
        fish = ContinuousFish(db, env_variables, 0.0, True, True, False)
    else:
        fish = Fish(db, env_variables, 0.0, True, True, False)
    return fish


def run_simulation(fish, actions, captures, duration=2000):
    energy_levels = []
    rewards = []
    for i in range(duration):
        reward = fish.take_action(actions[i])
        reward = fish.update_energy_level(reward, captures[i])
        energy_levels.append(fish.energy_level)
        rewards.append(reward)
        if fish.energy_level < 0:
            break

    fix, axs = plt.subplots()
    axs.plot(energy_levels)
    plt.show()
    # plt.plot(rewards)
    # plt.show()
    print(np.sum(rewards))


if __name__ == "__main__":
    learning_params, env_variables, n, b, c = load_assay_configuration_files("dqn_predator-1")
    env_variables["baseline_energy_use"] = 0.0002
    env_variables["a_scaling_energy_cost"] = 1.5e-04
    env_variables["i_scaling_energy_cost"] = 1.5e-04
    # env_variables["action_energy_use_scaling"] = "Nonlinear"
    # Modify env variables here.
    model = build_fish_model(env_variables, continuous=False)
    captures = np.random.choice([0, 1], size=1000, p=[1-0.00001, 0.00001]).astype(float)
    no_captures = np.zeros((2000))
    actions = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=2000)
    actions = [4 for i in range(2000)]
    # i = np.expand_dims((np.random.random_sample(1000) * 5), 1)
    # a = np.expand_dims((np.random.random_sample(1000)-0.5), axis=1)
    # actions = np.concatenate((i, a), axis=1)
    #actions[:, :] = 0
    # actions[:] = 6
    run_simulation(model, actions, no_captures)
