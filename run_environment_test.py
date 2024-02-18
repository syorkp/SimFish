
from Services.Testing.env_testing_service import EnvTestingService


if __name__ == "__main__":
    env_test = EnvTestingService("dqn_0", continuous_actions=False, display_board=True)
    env_test.episode_loop()
