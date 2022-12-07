import numpy as np

from Analysis.Calibration.Physics.test_environment import TestEnvironment




if __name__ == "__main__":
    env = TestEnvironment()
    env.run_prey_motion(num_steps=50)