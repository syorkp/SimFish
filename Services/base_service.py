import json
import cProfile
import shutil

import numpy as np
import os

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def delete_nv_folder():
    location = "./../../.nv"
    shutil.rmtree(location)

class BaseService:

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, new_simulation):

        if using_gpu:
            delete_nv_folder()

        self.monitor_performance = True  # TODO: make parameter
        if self.monitor_performance:
            self.profile = cProfile.Profile()
            self.profile.enable()

        super().__init__()

        # Name and location
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

    def create_session(self):
        print("Creating Session..")
        # TODO: Check is not breaking on GPU Usage (old Training and AssayServices did differently.

        if self.using_gpu:
            # options = tf.GPUOptions(per_process_gpu_memory_fraction=self.memory_fraction)
            # config = tf.ConfigProto(gpu_options=options)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = None

        if config:
            return tf.Session(config=config)
        else:
            return tf.Session()

    def load_configuration_files(self):
        with open(f"{self.current_configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{self.current_configuration_location}_env.json", 'r') as f:
            env = json.load(f)


        return params, env
