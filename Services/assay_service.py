import numpy as np
import json
from datetime import datetime

import tensorflow.compat.v1 as tf

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.controlled_stimulus_environment import ControlledStimulusEnvironment
from Environment.controlled_stimulus_environment_continuous import ControlledStimulusEnvironmentContinuous
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Services.base_service import BaseService
from Tools.make_gif import make_gif

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class AssayService(BaseService):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_environment, new_simulation, assays, set_random_seed,
                 assay_config_name, checkpoint):

        # Set random seed
        super().__init__(model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                         config_name, realistic_bouts, continuous_environment, new_simulation)

        print("AssayService Constructor called")

        if set_random_seed:
            np.random.seed(404)

        # Assay configuration and save location
        self.data_save_location = f"./Assay-Output/{self.model_id}"
        self.assay_configuration_id = assay_config_name
        self.checkpoint = checkpoint

        # Configuration
        self.current_configuration_location = f"./Configurations/Assay-Configs/{config_name}"
        self.learning_params, self.environment_params = self.load_configuration_files()
        self.assays = assays

        # Create environment so that network has access
        if self.continuous_actions:
            self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts, new_simulation, using_gpu)
        else:
            self.simulation = DiscreteNaturalisticEnvironment(self.environment_params, self.realistic_bouts, new_simulation, using_gpu)

        # Metadata
        self.episode_number = episode_number
        self.total_steps = total_steps

        # Output Data
        self.assay_output_data_format = None
        self.assay_output_data = []
        self.output_data = {}
        self.episode_summary_data = None

        # Hacky fix for h5py problem:
        self.last_position_dim = self.environment_params["prey_num"]
        self.stimuli_data = []

        # Placeholders overwritten by child class
        self.buffer = None
        self.ppo_version = None
        self.use_mu = False

    def _run(self):
        self.saver = tf.train.Saver(max_to_keep=5)
        self.init = tf.global_variables_initializer()
        checkpoint = tf.train.get_checkpoint_state(self.model_location)
        checkpoint_path = checkpoint.model_checkpoint_path
        if self.checkpoint is not None:
            checkpoint_path = self.model_location + f"/model-{self.checkpoint}.cptk"
        self.saver.restore(self.sess, checkpoint_path)
        print("Model loaded")
        for assay in self.assays:
            if assay["ablations"]:
                self.ablate_units(assay["ablations"])
            if self.environment_params["use_dynamic_network"]:
                if self.ppo_version is not None:
                    self.buffer.rnn_layer_names = self.actor_network.rnn_layer_names
                else:
                    self.buffer.rnn_layer_names = self.main_QN.rnn_layer_names

            self.save_frames = assay["save frames"]
            self.create_output_data_storage(assay)
            self.create_testing_environment(assay)
            self.perform_assay(assay)
            if assay["save stimuli"]:
                self.save_stimuli_data(assay)
        self.save_metadata()
        self.save_episode_data()

    def perform_assay(self, assay):
        self.assay_output_data_format = {key: None for key in assay["behavioural recordings"] + assay["network recordings"]}
        self.buffer.init_assay_recordings(assay["behavioural recordings"], assay["network recordings"])

        self.current_episode_max_duration = assay["duration"]
        if assay["use_mu"]:
            self.use_mu = True

        self._episode_loop()
        self.log_stimuli()

        if assay["save frames"]:
            make_gif(self.frame_buffer,
                     f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        self.frame_buffer = []

        if "reward assessments" in self.buffer.recordings:
            self.buffer.calculate_advantages_and_returns()
        self.buffer.save_assay_data(assay['assay id'], self.data_save_location, self.assay_configuration_id)
        self.buffer.reset()
        print(f"Assay: {assay['assay id']} Completed")

    def log_stimuli(self):
        stimuli = self.simulation.stimuli_information
        to_save = {}
        for stimulus in stimuli.keys():
            if stimuli[stimulus]:
                to_save[stimulus] = stimuli[stimulus]

        if to_save:
            self.stimuli_data.append(to_save)

    def create_testing_environment(self, assay):
        """
        Creates the testing environment as specified  by apparatus mode and given assays.
        :return:
        """
        if assay["stimulus paradigm"] == "Projection":
            if self.continuous_actions:
                self.simulation = ControlledStimulusEnvironmentContinuous(self.environment_params, assay["stimuli"],
                                                                self.realistic_bouts,
                                                                self.new_simulation,
                                                                self.using_gpu,
                                                                tethered=assay["Tethered"],
                                                                set_positions=assay["set positions"],
                                                                random=assay["random positions"],
                                                                moving=assay["moving"],
                                                                reset_each_step=assay["reset"],
                                                                reset_interval=assay["reset interval"],
                                                                background=assay["background"]
                                                                )
            else:
                self.simulation = ControlledStimulusEnvironment(self.environment_params, assay["stimuli"],
                                                                self.realistic_bouts,
                                                                self.new_simulation,
                                                                self.using_gpu,
                                                                tethered=assay["Tethered"],
                                                                set_positions=assay["set positions"],
                                                                random=assay["random positions"],
                                                                moving=assay["moving"],
                                                                reset_each_step=assay["reset"],
                                                                reset_interval=assay["reset interval"],
                                                                background=assay["background"]
                                                                )
        elif assay["stimulus paradigm"] == "Naturalistic":
            if self.continuous_actions:
                self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts,
                                                                    self.new_simulation, self.using_gpu,
                                                                    collisions=assay["collisions"])
            else:
                self.simulation = DiscreteNaturalisticEnvironment(self.environment_params, self.realistic_bouts,
                                                                self.new_simulation, self.using_gpu)

        else:
            if self.continuous_actions:
                self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts,
                                                                self.new_simulation, self.using_gpu)
            else:
                self.simulation = DiscreteNaturalisticEnvironment(self.environment_params, self.realistic_bouts,
                                                                self.new_simulation, self.using_gpu)

    def ablate_units(self, unit_indexes):
        # TODO: Will need to update for new network architecture.
        for unit in unit_indexes:
            if unit < 256:
                output = self.sess.graph.get_tensor_by_name('mainaw:0')
                new_tensor = output.eval()
                new_tensor[unit] = np.array([0 for i in range(10)])
                self.sess.run(tf.assign(output, new_tensor))
            else:
                output = self.sess.graph.get_tensor_by_name('mainvw:0')
                new_tensor = output.eval()
                new_tensor[unit - 256] = np.array([0])
                self.sess.run(tf.assign(output, new_tensor))

    def create_output_data_storage(self, assay):
        self.output_data = {key: [] for key in assay["behavioural recordings"] + assay["network recordings"]}
        self.output_data["step"] = []

    def save_episode_data(self):
        self.episode_summary_data = {
            "Prey Caught": self.simulation.prey_caught,
            "Predators Avoided": self.simulation.predator_attacks_avoided,
            "Sand Grains Bumped": self.simulation.sand_grains_bumped,
            "Steps Near Vegetation": self.simulation.steps_near_vegetation
        }
        with open(f"{self.data_save_location}/{self.assay_configuration_id}-summary_data.json", "w") as output_file:
            json.dump(self.episode_summary_data, output_file)
        self.episode_summary_data = None

    def save_stimuli_data(self, assay):
        with open(f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}-stimuli_data.json",
                  "w") as output_file:
            json.dump(self.stimuli_data, output_file)
        self.stimuli_data = []

    def save_metadata(self):
        metadata = {
            "Total Episodes": self.episode_number,
            "Total Steps": self.total_steps,
            "Assay Date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        with open(f"{self.data_save_location}/{self.assay_configuration_id}.json", "w") as output_file:
            json.dump(metadata, output_file)

    def save_assay_results(self, assay):
        """No longer used - saves data in JSON"""
        # Saves all the information from the assays in JSON format.
        if assay["save frames"]:
            make_gif(self.frame_buffer, f"{self.data_save_location}/{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'],
                     true_image=True)

        self.frame_buffer = []
        with open(f"{self.data_save_location}/{assay['assay id']}.json", "w") as output_file:
            json.dump(self.assay_output_data, output_file)
