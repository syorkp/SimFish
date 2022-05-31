import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_configuration_files
from Environment.Fish.fish import Fish
from Environment.Fish.continuous_fish import ContinuousFish
from Tools.drawing_board_new import NewDrawingBoard


def build_fish_model(env_variables, continuous):

    db = NewDrawingBoard(env_variables["width"],
                         env_variables["height"],
                         env_variables["decay_rate"],
                         env_variables["uv_photoreceptor_rf_size"],
                         False,
                         False)
    if continuous:
        fish = ContinuousFish(db, env_variables, 0.0, True, True, False)
    else:
        fish = Fish(db, env_variables, 0.0, True, True, False)
    return fish


def run_simulation(fish, actions, captures, duration=1000):
    energy_levels = []
    rewards = []
    for i in range(duration):
        reward = fish.take_action(actions[i])
        reward = fish.update_energy_level(reward, captures[i])
        energy_levels.append(fish.energy_level)
        rewards.append(reward)
        if fish.energy_level < 0:
            break

    plt.plot(energy_levels)
    plt.show()
    plt.plot(rewards)
    plt.show()
    print(np.sum(rewards))


learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_15-1")
env_variables["baseline_decrease"] = 0.0015
env_variables["ci"] = 0.0003
env_variables["ca"] = 0.0003
# Modify env variables here.
model = build_fish_model(env_variables, continuous=False)
captures = np.random.choice([0, 1], size=1000, p=[1-0.001, 0.001]).astype(float)
no_captures = np.zeros((1000))
actions = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=1000)
# actions[:] = 6
run_simulation(model, actions, no_captures)
