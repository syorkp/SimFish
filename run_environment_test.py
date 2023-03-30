from Services.Testing.env_testing_service import EnvTestingService
import numpy as np


if __name__ == "__main__":
    env_test = EnvTestingService("epsilon_proj_3", continuous_actions=False)
    # env_test.controlled_episode_loop()
    env_test.run_full_episode(300, random_position=True)
