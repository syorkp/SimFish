
from Services.Testing.env_testing_service import EnvTestingService


if __name__ == "__main__":
    env_test = EnvTestingService("epsilon_new", continuous_actions=False)
    env_test.episode_loop()
