import copy

import numpy as np

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Analysis.Video.behaviour_video_construction import draw_episode
from Analysis.load_data import load_data
from Buffers.DQN.dqn_assay_buffer import DQNAssayBuffer
from Services.assay_service import AssayService
from Services.DQN.base_dqn import BaseDQN


tf.logging.set_verbosity(tf.logging.ERROR)


def assay_target(trial, total_steps, episode_number, memory_fraction):
    if "Using GPU" in trial:
        using_gpu = trial["Using GPU"]
    else:
        using_gpu = True

    if "Checkpoint" in trial:
        checkpoint = trial["Checkpoint"]
    else:
        checkpoint = None

    if "interventions" in trial:
        interventions = trial["interventions"]
    else:
        interventions = None

    if "set random seed" in trial:
        set_random_seed = trial["set random seed"]
    else:
        set_random_seed = False

    if "behavioural recordings" in trial:
        behavioural_recordings = trial["behavioural recordings"]
    else:
        behavioural_recordings = None

    if "network recordings" in trial:
        network_recordings = trial["network recordings"]
    else:
        network_recordings = None

    # Handling for when using split assay version.
    modification = None
    split_event = None
    if trial["Run Mode"] == "Split-Assay":
        split_event = trial["Split Event"]
        if "Run Index" not in trial:
            run_version = "Original"
        else:
            modification = trial["Modification"]
            run_version = trial["Run Index"]
    else:
        run_version = None

    service = DQNAssayService(model_name=trial["Model Name"],
                              trial_number=trial["Trial Number"],
                              total_steps=total_steps,
                              episode_number=episode_number,
                              using_gpu=using_gpu,
                              memory_fraction=memory_fraction,
                              config_name=trial["Environment Name"],
                              assays=trial["Assays"],
                              set_random_seed=set_random_seed,
                              assay_config_name=trial["Assay Configuration Name"],
                              checkpoint=checkpoint,
                              behavioural_recordings=behavioural_recordings,
                              network_recordings=network_recordings,
                              interventions=interventions,
                              run_version=run_version,
                              split_event=split_event,
                              modification=modification
                              )

    service.run()


class DQNAssayService(AssayService, BaseDQN):

    def __init__(self, model_name, trial_number, total_steps, episode_number, using_gpu, memory_fraction,
                 config_name, assays, set_random_seed, assay_config_name, checkpoint, behavioural_recordings,
                 network_recordings, interventions, run_version, split_event, modification):
        """
        Runs a set of assays provided by the run configuration.
        """

        super().__init__(model_name=model_name,
                         trial_number=trial_number,
                         total_steps=total_steps,
                         episode_number=episode_number,
                         using_gpu=using_gpu,
                         memory_fraction=memory_fraction,
                         config_name=config_name,
                         continuous_environment=False,
                         assays=assays,
                         set_random_seed=set_random_seed,
                         assay_config_name=assay_config_name,
                         checkpoint=checkpoint,
                         behavioural_recordings=behavioural_recordings,
                         network_recordings=network_recordings,
                         interventions=interventions,
                         run_version=run_version,
                         split_event=split_event,
                         modification=modification
                         )

        # Hacky fix for h5py problem:
        self.last_position_dim = self.environment_params["prey_num"]
        self.stimuli_data = []
        self.output_data = {}

        self.buffer = DQNAssayBuffer()

    def run(self):
        sess = self.create_session()
        with sess as self.sess:
            self.create_network()
            self.init_states()
            AssayService._run(self)

    def perform_assay(self, assay, sediment=None, energy_state=None):
        """Perform the specified assay - episode_loop for assay mode."""

        if self.rnn_input is not None:
            rnn_state = copy.copy(self.rnn_input[0])
            rnn_state_ref = copy.copy(self.rnn_input[1])
        else:
            rnn_state = copy.copy(self.init_rnn_state)
            rnn_state_ref = copy.copy(self.init_rnn_state_ref)

        if self.run_version == "Original-Completion" or self.run_version == "Modified-Completion":
            print("Loading Simulation")
            a = self.load_simulation(sediment, energy_state, rnn_state, rnn_state_ref)
            a = a[0]
            self.step_number = len(self.buffer.internal_state_buffer)

        else:
            self.simulation.reset()
            a = 0
            self.step_number = 0

        o, r, internal_state, d, full_masked_image = self.simulation.simulation_step(action=a)

        efference_copy = [[a, self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle]]

        while self.step_number < assay["duration"]:
            self.step_number += 1

            # Deal with interventions
            if self.interruptions:
                o, efference_copy, internal_state = self.perform_interruptions(o, efference_copy, internal_state)

            self.previous_action = a

            o, a, r, internal_state, o1, d, rnn_state, rnn_state_ref = self.step_loop(o=o,
                                                                                      internal_state=internal_state,
                                                                                      a=efference_copy,
                                                                                      rnn_state=rnn_state,
                                                                                      rnn_state_ref=rnn_state_ref
                                                                                      )
            o = o1

            if d:
                if self.run_version == "Original":
                    if self.simulation.switch_step is not None:
                        self.buffer.switch_step = self.simulation.switch_step
                    else:
                        # If no split occurs, return without saving data.
                        print("No split occurred, as condition never met. Returning without saving data.")
                        return False

                break

            efference_copy = [a]

        # Catch in case assay duration completes without the split being made.
        if self.run_version == "Original":
            if self.simulation.switch_step is not None:
                self.buffer.switch_step = self.simulation.switch_step
            else:
                # If no split occurs, return without saving data.
                print("No split occurred, as condition never met. Returning without saving data.")
                return False

        if self.environment_params["salt"]:
            salt_location = self.simulation.salt_location
        else:
            salt_location = None

        # TODO: Temp change here
        if self.using_gpu:
            background = None  #self.simulation.board.global_sediment_grating.get()[:, :, 0]
        else:
            background = self.simulation.board.global_sediment_grating[:, :, 0]

        self.buffer.save_assay_data(assay['assay id'], self.data_save_location, self.assay_configuration_id,
                                    self.internal_state_order, sediment=background,
                                    salt_location=salt_location)
        self.log_stimuli()
        self.buffer.reset()
        if assay["save frames"]:
            episode_data = load_data(f"{self.model_name}-{self.model_number}", self.assay_configuration_id,
                                     assay['assay id'], training_data=False)
            draw_episode(episode_data, self.config_name, f"{self.model_name}-{self.model_number}", self.continuous_actions,
                         save_id=f"{self.assay_configuration_id}-{assay['assay id']}")

        print(f"Assay: {assay['assay id']} Completed")
        print("")
        return True

    def step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref):
        return BaseDQN.assay_step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref)

    def load_simulation(self, sediment, energy_state, rnn_state, rnn_state_ref):
        """Load the simulation in a given state - used for split assay mode."""
        o = self.simulation.load_simulation(self.buffer, sediment, energy_state)
        internal_state = self.buffer.internal_state_buffer[-1]

        a = self.buffer.action_buffer[-1]
        a = np.array([a, self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle])

        if self.run_version == "Modified-Completion":
            self.simulation.make_modification()

        a, updated_rnn_state, rnn2_state = \
            self.sess.run(
                [self.main_QN.predict, self.main_QN.rnn_state_shared, self.main_QN.rnn_state_ref],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: np.expand_dims(a, 0),
                           self.main_QN.train_length: 1,
                           self.main_QN.rnn_state_in: rnn_state,
                           self.main_QN.rnn_state_in_ref: rnn_state_ref,

                           self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0,
                           })

        return a
