import json
import cProfile

import numpy as np
import os

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class BaseService:

    def __init__(self, model_name, trial_number, total_steps, episode_number, using_gpu, memory_fraction,
                 config_name, continuous_actions, monitor_performance=False):

        self.monitor_performance = monitor_performance
        if self.monitor_performance:
            self.profile = cProfile.Profile()
            self.profile.enable()

        super().__init__()

        # Name and location
        self.model_name = model_name
        self.model_number = trial_number
        self.model_id = f"{model_name}-{trial_number}"
        self.model_location = f"./Training-Output/{self.model_id}"

        # Configuration
        self.config_name = config_name
        self.current_configuration_location = None

        # Computation Parameters
        self.using_gpu = using_gpu
        self.memory_fraction = memory_fraction

        # Networks Parameters
        self.saver = None
        self.init = None
        self.sess = None

        # Simulation
        self.simulation = None
        self.step_number = 0
        self.continuous_actions = continuous_actions

        # Maintain trial variables
        if episode_number is not None:
            self.episode_number = episode_number + 1
        else:
            self.episode_number = 0

        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = 0

        if not self.using_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Placeholder Attributes
        if not hasattr(self, "environment_params"):
            self.environment_params = None
        if not hasattr(self, "episode_buffer"):
            self.episode_buffer = None
        if not hasattr(self, "buffer"):
            self.buffer = None
        if not hasattr(self, "episode_loop"):
            self.episode_loop = None
        if not hasattr(self, "network"):
            self.network = None
        if not hasattr(self, "main_QN"):
            self.main_QN = None
        if not hasattr(self, "_episode_loop"):
            self._episode_loop = None

    def create_session(self):
        """Returns a tf.Session() object."""

        print("Creating Session..")

        if self.using_gpu:
            print("Using GPU")
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = None

        if config:
            print("Using GPU Confirmed")
            return tf.Session(config=config)
        else:
            return tf.Session()

    def load_configuration_files(self):
        """Load configuration files."""
        with open(f"{self.current_configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{self.current_configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return params, env

    def get_internal_state_order(self):
        """Returns a list, indicating the order of variables in the internal state provided to the network."""
        internal_state_order = []
        if self.environment_params['in_light']:
            internal_state_order.append("in_light")
        if self.environment_params['stress']:
            internal_state_order.append("stress")
        if self.environment_params['energy_state']:
            internal_state_order.append("energy_state")
        if self.environment_params['salt']:
            internal_state_order.append("salt")
        internal_state_order = ['a1', 'a2', 'a3']
        return internal_state_order

    def get_feature_positions(self):
        """Returns the positions of all environmental features for logging."""
        if len(self.simulation.sand_grain_bodies) > 0:
            sand_grain_positions = [self.simulation.sand_grain_bodies[i].position for i, b in
                                    enumerate(self.simulation.sand_grain_bodies)]
            sand_grain_positions = [[i[0], i[1]] for i in sand_grain_positions]
        else:
            sand_grain_positions = [[15000, 15000]]

        if self.simulation.prey_bodies:
            prey_positions = [prey.position for prey in self.simulation.prey_bodies]
            prey_positions = np.array([[i[0], i[1]] for i in prey_positions])
        else:
            prey_positions = np.array([[15000, 15000]])

        if self.simulation.predator_body is not None:
            predator_position = self.simulation.predator_body.position
            predator_position = np.array([predator_position[0], predator_position[1]])
        else:
            predator_position = np.array([15000, 15000])

        return sand_grain_positions, prey_positions, predator_position

    def log_data(self, efference_copy, a):
        """Log data from an episode."""
        sand_grain_positions, prey_positions, predator_position = self.get_feature_positions()
        prey_orientations = np.array([p.angle for p in self.simulation.prey_bodies]).astype(np.float32)

        try:
            predator_orientation = self.simulation.predator_body.angle
        except:
            predator_orientation = 0

        prey_ages = np.array(self.simulation.prey_ages)
        prey_gait = np.array(self.simulation.paramecia_gaits)

        prey_identifiers = np.array(self.simulation.prey_identifiers)

        self.buffer.save_environmental_positions(action=a,
                                                 fish_position=self.simulation.fish.body.position,
                                                 prey_consumed=self.simulation.prey_consumed_this_step,
                                                 predator_present=self.simulation.predator_body,
                                                 prey_positions=prey_positions,
                                                 predator_position=predator_position,
                                                 sand_grain_positions=sand_grain_positions,
                                                 fish_angle=self.simulation.fish.body.angle,
                                                 salt_health=self.simulation.fish.salt_health,
                                                 efference_copy=efference_copy,
                                                 prey_orientation=prey_orientations,
                                                 predator_orientation=predator_orientation,
                                                 prey_age=prey_ages,
                                                 prey_gait=prey_gait,
                                                 prey_identifiers=prey_identifiers
                                                 )

