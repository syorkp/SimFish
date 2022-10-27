import json
import cProfile
import shutil

import numpy as np
import os

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def delete_nv_folder():
    print(os.getcwd())
    location = "./../../../../home/zcbtspi/.nv"
    if os.path.isdir(location):
        print(f"Correct dir, removing {location}")
        shutil.rmtree(location)
        if os.path.isdir(location):
            print("Failed...")
    # location = "./../../.nv"
    # if os.path.isdir(location):
    #     print(f"Correct dir, removing {location}")
    #     shutil.rmtree(location)
    #     if os.path.isdir(location):
    #         print("Failed...")
    # location = "./../.nv"
    # if os.path.isdir(location):
    #     print("Correct dir, removing")
    #     shutil.rmtree(location)
    #     if os.path.isdir(location):
    #         print("Failed...")


class BaseService:

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, new_simulation, monitor_performance=False):

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
        self.monitor_gpu = monitor_gpu
        self.memory_fraction = memory_fraction

        # Networks Parameters
        self.saver = None
        self.init = None
        self.sess = None

        # Simulation
        self.simulation = None
        self.step_number = 0
        self.continuous_actions = continuous_actions
        self.realistic_bouts = realistic_bouts

        # Data
        self.save_frames = False
        self.frame_buffer = []

        # Maintain trial variables
        if episode_number is not None:
            self.episode_number = episode_number + 1
        else:
            self.episode_number = 0

        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = 0

        # Switchover at start of phase 1
        self.new_simulation = new_simulation

        if not self.using_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Add attributes only if don't exist yet (prevents errors thrown).
        if not hasattr(self, "environment_params"):
            self.environment_params = None
        if not hasattr(self, "last_position_dim"):
            self.last_position_dim = None
        if not hasattr(self, "episode_buffer"):
            self.episode_buffer = None
        if not hasattr(self, "buffer"):
            self.buffer = None
        if not hasattr(self, "episode_loop"):
            self.episode_loop = None
        if not hasattr(self, "actor_network"):
            self.actor_network = None
        if not hasattr(self, "main_QN"):
            self.main_QN = None
        if not hasattr(self, "_episode_loop"):
            self._episode_loop = None

    def create_session(self):
        print("Creating Session..")

        if self.using_gpu:
            # options = tf.GPUOptions(per_process_gpu_memory_fraction=self.memory_fraction)
            # config = tf.ConfigProto(gpu_options=options)
            print("Using GPU")
            try:
                delete_nv_folder()
            except FileNotFoundError:
                pass
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
        with open(f"{self.current_configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{self.current_configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return params, env

    def get_internal_state_order(self):
        internal_state_order = []
        if self.environment_params['in_light']:
            internal_state_order.append("in_light")
        if self.environment_params['hunger']:
            internal_state_order.append("hunger")
        if self.environment_params['stress']:
            internal_state_order.append("stress")
        if self.environment_params['energy_state']:
            internal_state_order.append("energy_state")
        if self.environment_params['salt']:
            internal_state_order.append("salt")
        return internal_state_order

    def get_positions(self):
        """Should be here as is used in both training and assay services."""
        if len(self.simulation.sand_grain_bodies) > 0:
            sand_grain_positions = [self.simulation.sand_grain_bodies[i].position for i, b in
                                    enumerate(self.simulation.sand_grain_bodies)]
            sand_grain_positions = [[i[0], i[1]] for i in sand_grain_positions]
        else:
            sand_grain_positions = [[10000, 10000]]

        if self.simulation.prey_bodies:
            prey_positions = [prey.position for prey in self.simulation.prey_bodies]
            prey_positions = [[i[0], i[1]] for i in prey_positions]
            while True:
                if len(prey_positions) < self.last_position_dim:
                    prey_positions = np.append(prey_positions, [[10000, 10000]], axis=0)
                else:
                    break

            self.last_position_dim = len(prey_positions)

        else:
            prey_positions = np.array([[10000, 10000]])

        if self.simulation.predator_body is not None:
            predator_position = self.simulation.predator_body.position
            predator_position = np.array([predator_position[0], predator_position[1]])
        else:
            predator_position = np.array([10000, 10000])

        if self.simulation.vegetation_bodies is not None:
            vegetation_positions = [self.simulation.vegetation_bodies[i].position for i, b in
                                    enumerate(self.simulation.vegetation_bodies)]
            vegetation_positions = [[i[0], i[1]] for i in vegetation_positions]
        else:
            vegetation_positions = [[10000, 10000]]

        return sand_grain_positions, prey_positions, predator_position, vegetation_positions
