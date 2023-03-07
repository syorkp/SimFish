import numpy as np
import json
from datetime import datetime
import copy
import h5py

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.controlled_stimulus_environment import ControlledStimulusEnvironment
from Environment.controlled_stimulus_environment_continuous import ControlledStimulusEnvironmentContinuous
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Services.base_service import BaseService


tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class AssayService(BaseService):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, continuous_environment, assays, set_random_seed, assay_config_name, checkpoint,
                 behavioural_recordings, network_recordings, interventions, run_version, split_event, modification):

        super().__init__(model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                         config_name, continuous_environment)

        # Assay configuration and save location
        self.data_save_location = f"./Assay-Output/{self.model_id}"
        self.assay_configuration_id = assay_config_name
        self.checkpoint = checkpoint

        # Configuration
        self.current_configuration_location = f"./Configurations/Assay-Configs/{config_name}"
        self.learning_params, self.environment_params = self.load_configuration_files()

        # Handling for split assay version
        if run_version == "Original-Completion":
            print("Running for completion of original")
            set_random_seed = True
        elif run_version == "Modified-Completion":
            print("Running for completion of modified")
            set_random_seed = True
            for assay in assays:
                assay["assay id"] += "-Mod"
        elif run_version == "Original":
            print("Running for pre-split original")
        elif run_version is None:
            pass
        else:
            print("Incorrectly specified.")

        self.run_version = run_version
        self.modification = modification
        self.split_event = split_event

        self.assays = self.expand_assays(assays)

        # Set random seed
        if set_random_seed:
            np.random.seed(404)

        # Create environment so that network has access
        if self.continuous_actions:
            self.simulation = ContinuousNaturalisticEnvironment(env_variables=self.environment_params,
                                                                using_gpu=using_gpu,
                                                                run_version=run_version,
                                                                split_event=split_event,
                                                                modification=modification,
                                                                )
        else:
            self.simulation = DiscreteNaturalisticEnvironment(env_variables=self.environment_params,
                                                              using_gpu=using_gpu,
                                                              run_version=run_version,
                                                              split_event=split_event,
                                                              modification=modification,
                                                              )

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

        self.internal_state_order = self.get_internal_state_order()

        self.preset_energy_state = None
        self.efference_copy_interruptions = None
        self.visual_interruptions = None
        self.previous_action = None
        self.relocate_fish = None
        self.salt_interruptions = None
        self.in_light_interruptions = None
        self.interruptions = False
        self.rnn_input = None

        self.behavioural_recordings = behavioural_recordings
        self.network_recordings = network_recordings
        self.interventions = interventions

    def implement_interventions(self):
        if self.interventions is not None:
            if "visual_interruptions" in self.interventions.keys():
                self.visual_interruptions = self.interventions["visual_interruptions"]
            if "efference_copy_interruptions" in self.interventions.keys():
                self.efference_copy_interruptions = self.interventions["efference_copy_interruptions"]
            if "preset_energy_state" in self.interventions.keys():
                self.preset_energy_state = self.interventions["preset_energy_state"]
            if "relocate_fish" in self.interventions.keys():
                self.relocate_fish = self.interventions["relocate_fish"]
            if "in_light_interruptions" in self.interventions.keys():
                self.in_light_interruptions = self.interventions["in_light_interruptions"]
            if "salt_interruptions" in self.interventions.keys():
                self.salt_interruptions = self.interventions["salt_interruptions"]
            if "ablations" in self.interventions.keys():
                self.ablate_units(self.interventions["ablations"])
            if "rnn_input" in self.interventions.keys():
                self.rnn_input = self.interventions["rnn_input"]
            self.interruptions = True
        else:
            self.interruptions = False

    def _run(self):
        self.saver = tf.train.Saver(max_to_keep=5)
        self.init = tf.global_variables_initializer()
        checkpoint = tf.train.get_checkpoint_state(self.model_location)
        checkpoint_path = checkpoint.model_checkpoint_path
        if self.checkpoint is not None:
            checkpoint_path = self.model_location + f"/model-{self.checkpoint}.cptk"
        self.saver.restore(self.sess, checkpoint_path)
        print("Model loaded")

        self.implement_interventions()

        if self.ppo_version is not None:
            self.buffer.rnn_layer_names = self.network.rnn_layer_names
        else:
            self.buffer.rnn_layer_names = self.main_QN.rnn_layer_names

        for assay in self.assays:

            # Init recordings
            self.create_output_data_storage(self.behavioural_recordings, self.network_recordings)
            self.buffer.init_assay_recordings(self.behavioural_recordings, self.network_recordings)

            self.learning_params, self.environment_params = self.load_configuration_files()

            self.save_frames = assay["save frames"]

            self.create_testing_environment(assay)

            if self.run_version == "Original-Completion" or self.run_version == "Modified-Completion":
                sediment, energy_state = self.load_assay_buffer(assay)

                # End in the event first trial had no end.
                if sediment is False:
                    return
            else:
                sediment, energy_state = None, None

            while True:
                complete = self.perform_assay(assay, sediment=sediment, energy_state=energy_state)
                if complete:
                    break

            if assay["save stimuli"]:
                self.save_stimuli_data(assay)

            self.visual_interruptions = None
            self.efference_copy_interruptions = None
            self.preset_energy_state = None
            self.relocate_fish = None

        self.save_metadata()
        self.save_episode_data()

    def expand_assays(self, assays):
        """Utilitiy function to add in repeats of any assays and bring them into the standard of the program (while
         allowing immediate simplifications to be made)"""
        new_assays = []
        for assay in assays:
            for i in range(1, assay["repeats"]+1):
                new_assay_format = copy.copy(assay)
                new_assay_format["assay id"] = f"{new_assay_format['assay id']}-{i}"
                new_assays.append(new_assay_format)
        return new_assays

    def load_assay_buffer(self, assay):
        print("Loading Assay Buffer")
        # Get the assay id of the base trial (without mod)
        assay_id = assay["assay id"].replace("-Mod", "")
        file = h5py.File(f"{self.data_save_location}/{self.assay_configuration_id}.h5", "r")
        g = file.get(assay_id)

        data = {key: np.array(g.get(key)) for key in g.keys()}

        try:
            self.buffer.switch_step = data["switch_step"]
            self.total_steps = data["switch_step"]
        except KeyError:
            print("Trial had no switch.")
            return False, False

        # Impose buffer
        if self.continuous_actions:
            self.buffer.action_buffer = np.concatenate((np.expand_dims(data["impulse"], 1), np.expand_dims(data["angle"], 1)), axis=1).tolist()
        else:
            self.buffer.action_buffer = data["action"].tolist()

        self.buffer.observation_buffer = data["observation"].tolist()
        self.buffer.reward_buffer = data["reward"].tolist()
        self.buffer.internal_state_buffer = [np.array([internal_state]) for internal_state in data["internal_state"].tolist()]
        self.buffer.efference_copy_buffer = data["efference_copy"].tolist()

        self.buffer.rnn_state_buffer = [np.array([([timepoint[0]], [timepoint[1]])]) for timepoint in data["rnn_state"]]
        self.buffer.rnn_state_ref_buffer = [np.array([([timepoint[0]], [timepoint[1]])]) for timepoint in data["rnn_state_ref"]]

        self.buffer.fish_position_buffer = data["fish_position"].tolist()
        self.buffer.fish_angle_buffer = data["angle"].tolist()
        self.buffer.predator_position_buffer = data["predator_positions"].tolist()
        self.buffer.salt_health_buffer = data["salt_health"].tolist()
        self.buffer.prey_positions_buffer = data["prey_positions"].tolist()
        self.buffer.sand_grain_position_buffer = data["sand_grain_positions"].tolist()
        self.buffer.salt_location = data["salt_location"].tolist()
        self.buffer.prey_consumed_buffer = data["consumed"].tolist()

        energy_state = data["energy_state"][-1]

        self.buffer.prey_orientations_buffer = data["prey_orientations"].tolist()
        self.buffer.predator_orientation_buffer = data["predator_orientation"].tolist()
        self.buffer.prey_age_buffer = data["prey_ages"].tolist()
        self.buffer.prey_gait_buffer = data["prey_gaits"].tolist()

        # Load RNN states to model.
        num_rnns = np.array(self.buffer.rnn_state_buffer).shape[2] / 2
        self.init_rnn_state = tuple(
            (np.array(data["rnn_state"][-2:-1, shape]),
             np.array(data["rnn_state"][-2:-1, shape])) for shape in
            range(0, int(num_rnns), 2))
        self.init_rnn_state_ref = tuple(
            (np.array(data["rnn_state_ref"][-2:-1, shape]),
             np.array(data["rnn_state_ref"][-2:-1, shape])) for shape in
            range(0, int(num_rnns), 2))

        # Impose sediment (red2 channel).
        return data["sediment"], energy_state

    def log_stimuli(self):
        stimuli = self.simulation.stimuli_information
        to_save = {}
        for stimulus in stimuli.keys():
            if stimuli[stimulus]:
                to_save[stimulus] = stimuli[stimulus]

        if to_save:
            self.stimuli_data.append(to_save)
        self.simulation.stimuli_information = {}

    def create_testing_environment(self, assay):
        """
        Creates the testing environment as specified  by apparatus mode and given assays.
        :return:
        """

        if assay["stimulus paradigm"] == "Projection":
            if self.continuous_actions:
                self.simulation = ControlledStimulusEnvironmentContinuous(env_variables=self.environment_params,
                                                                          stimuli=assay["stimuli"],
                                                                          using_gpu=self.using_gpu,
                                                                          tethered=assay["Tethered"],
                                                                          set_positions=assay["set positions"],
                                                                          random=assay["random positions"],
                                                                          moving=assay["moving"],
                                                                          reset_each_step=assay["reset"],
                                                                          reset_interval=assay["reset interval"],
                                                                          sediment=assay["sediment"],
                                                                          assay_all_details=assay
                                                                          )
            else:
                self.simulation = ControlledStimulusEnvironment(env_variables=self.environment_params,
                                                                stimuli=assay["stimuli"],
                                                                using_gpu=self.using_gpu,
                                                                tethered=assay["Tethered"],
                                                                set_positions=assay["set positions"],
                                                                random=assay["random positions"],
                                                                moving=assay["moving"],
                                                                reset_each_step=assay["reset"],
                                                                reset_interval=assay["reset interval"],
                                                                sediment=assay["sediment"],
                                                                assay_all_details=assay,
                                                                )
        elif assay["stimulus paradigm"] == "Naturalistic":
            if self.continuous_actions:
                self.simulation = ContinuousNaturalisticEnvironment(env_variables=self.environment_params,
                                                                    using_gpu=self.using_gpu,
                                                                    relocate_fish=self.relocate_fish,
                                                                    run_version=self.run_version,
                                                                    split_event=self.split_event,
                                                                    modification=self.modification,
                                                                    )
            else:
                self.simulation = DiscreteNaturalisticEnvironment(env_variables=self.environment_params,
                                                                  using_gpu=self.using_gpu,
                                                                  relocate_fish=self.relocate_fish,
                                                                  run_version=self.run_version,
                                                                  split_event=self.split_event,
                                                                  modification=self.modification,
                                                                  )

        else:
            if self.continuous_actions:
                self.simulation = ContinuousNaturalisticEnvironment(env_variables=self.environment_params,
                                                                    using_gpu=self.using_gpu,
                                                                    relocate_fish=self.relocate_fish,
                                                                    )
            else:
                self.simulation = DiscreteNaturalisticEnvironment(env_variables=self.environment_params,
                                                                  using_gpu=self.using_gpu,
                                                                  relocate_fish=self.relocate_fish,
                                                                  )

    def ablate_units(self, ablated_layers):

        for layer in ablated_layers.keys():
            if layer == "rnn_weights":
                target = "main_rnn/lstm_cell/kernel:0"
            elif layer == "Advantage":
                target = "mainaw:0"
            else:
                target = 'mainvw:0'

            original_matrix = self.sess.graph.get_tensor_by_name(target)
            self.sess.run(tf.assign(original_matrix, ablated_layers[layer]))
            print(f"Ablated {layer}")

    def create_output_data_storage(self, behavioural_recordings, network_recordings):
        self.output_data = {key: [] for key in behavioural_recordings + network_recordings}
        self.output_data["step"] = []

    def save_episode_data(self):
        self.episode_summary_data = {
            "Prey Caught": self.simulation.prey_caught,
            "Predators Avoided": self.simulation.predator_attacks_avoided,
            "Sand Grains Bumped": self.simulation.sand_grains_bumped,
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
