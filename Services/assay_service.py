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
                 config_name, realistic_bouts, continuous_environment, new_simulation, assays, set_random_seed, assay_config_name):

        # Set random seed
        super().__init__(model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                         config_name, realistic_bouts, continuous_environment, new_simulation)

        print("AssayService Constructor called")

        if set_random_seed:
            np.random.seed(404)

        # Assay configuration and save location
        self.data_save_location = f"./Assay-Output/{self.model_id}"
        self.assay_configuration_id = assay_config_name

        # Configuration
        self.current_configuration_location = f"./Configurations/Assay-Configs/{config_name}"
        self.learning_params, self.environment_params = self.load_configuration_files()
        self.assays = assays

        # Create environment so that network has access
        if self.continuous_actions:
            self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts)
        else:
            self.simulation = DiscreteNaturalisticEnvironment(self.environment_params, self.realistic_bouts)

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

    def _run(self):
        self.saver = tf.train.Saver(max_to_keep=5)
        self.init = tf.global_variables_initializer()
        checkpoint = tf.train.get_checkpoint_state(self.model_location)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        print("Model loaded")
        for assay in self.assays:
            if assay["ablations"]:
                self.ablate_units(assay["ablations"])
            if self.new_simulation:
                self.buffer.rnn_layer_names = self.actor_network.rnn_layer_names
            self.save_frames = assay["save frames"]
            self.create_output_data_storage(assay)
            self.create_testing_environment(assay)
            self.perform_assay(assay)
            if assay["save stimuli"]:
                self.save_stimuli_data(assay)
        self.save_metadata()
        self.save_episode_data()

    def perform_assay(self, assay):
        self.assay_output_data_format = {key: None for key in assay["recordings"]}
        self.buffer.recordings = assay["recordings"]
        self.current_episode_max_duration = assay["duration"]

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

    def get_positions(self):
        if not self.simulation.sand_grain_bodies:
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

    def create_output_data_storage(self, assay):
        self.output_data = {key: [] for key in assay["recordings"]}
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
