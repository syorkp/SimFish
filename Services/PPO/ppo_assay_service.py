import numpy as np
# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import copy

from Buffers.PPO.ppo_buffer import PPOBuffer

from Services.PPO.continuous_ppo import ContinuousPPO
from Services.assay_service import AssayService
from Analysis.Video.behaviour_video_construction import draw_episode
from Analysis.load_data import load_data

tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_assay_target_continuous(trial, total_steps, episode_number, memory_fraction):
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

    service = PPOAssayServiceContinuous(model_name=trial["Model Name"],
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


class PPOAssayServiceContinuous(AssayService, ContinuousPPO):

    def __init__(self, model_name, trial_number, total_steps, episode_number, using_gpu, memory_fraction,
                 config_name, assays, set_random_seed, assay_config_name,
                 checkpoint, behavioural_recordings, network_recordings, interventions, run_version, split_event,
                 modification):
        """
        Runs a set of assays provided by the run configuration.
        """
        # Set random seed
        super().__init__(model_name=model_name,
                         trial_number=trial_number,
                         total_steps=total_steps,
                         episode_number=episode_number,
                         using_gpu=using_gpu,
                         memory_fraction=memory_fraction,
                         config_name=config_name,
                         continuous_environment=True,
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

        # Buffer for saving results of assay
        self.buffer = PPOBuffer(gamma=0.99,
                                lmbda=0.9,
                                batch_size=self.learning_params["batch_size"],
                                train_length=self.learning_params["trace_length"],
                                assay=True,
                                debug=False,
                                )

        self.ppo_version = ContinuousPPO
        self.epsilon = self.learning_params["startE"] / 2
        self.epsilon_greedy = self.learning_params["epsilon_greedy"]
        self.last_position_dim = self.environment_params["prey_num"]

    def run(self):
        sess = self.create_session()
        with sess as self.sess:
            self.create_network()
            self.init_states()
            AssayService._run(self)

    def perform_assay(self, assay, sediment=None, energy_state=None):
        """Perform assay - used instead of episode loop"""

        self.update_sigmas()

        if self.rnn_input is not None:
            rnn_state = copy.copy(self.rnn_input[0])
            rnn_state_ref = copy.copy(self.rnn_input[1])
        else:
            rnn_state = copy.copy(self.init_rnn_state)
            rnn_state_ref = copy.copy(self.init_rnn_state_ref)

        self.current_episode_max_duration = assay["duration"]
        if assay["use_mu"]:
            self.use_mu = True

        if self.run_version == "Original-Completion" or self.run_version == "Modified-Completion":
            print("Loading Simulation")

            o = self.simulation.load_simulation(self.buffer, sediment, energy_state)
            self.simulation.prey_identifiers = copy.copy(self.buffer.prey_identifiers_buffer[-1])
            self.simulation.total_prey_created = int(max([max(p_i) for p_i in self.buffer.prey_identifiers_buffer]) + 1)

            internal_state = self.buffer.internal_state_buffer[-1]
            a = self.buffer.action_buffer[-1]

            a = np.array(a + [self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle])

            if self.run_version == "Modified-Completion":
                self.simulation.make_modification()

            impulse, angle, updated_rnn_state, updated_rnn_state_ref, mu_i, mu_a, \
            si = \
                self.sess.run(
                    [self.network.impulse_output, self.network.angle_output, self.network.rnn_state_shared,
                     self.network.rnn_state_ref,
                     self.network.mu_impulse_combined,
                     self.network.mu_angle_combined,
                     self.network.sigma_action,
                     ],
                    feed_dict={self.network.observation: o,
                               self.network.internal_state: internal_state,
                               self.network.prev_actions: np.expand_dims(a, 0),
                               self.network.train_length: 1,
                               self.network.batch_size: 1,
                               self.network.rnn_state_in: rnn_state,
                               self.network.rnn_state_in_ref: rnn_state_ref,
                               self.network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.network.sigma_angle_combined_proto: self.angle_sigma,
                               self.network.entropy_coefficient: self.learning_params["lambda_entropy"],

                               })
            a = [mu_i[0][0], mu_a[0][0]]
            self.step_number = len(self.buffer.internal_state_buffer)

        else:
            self.simulation.reset()
            a = [4.0, 0.0]
            self.step_number = 0

        if self.assay or self.just_trained:
            self.buffer.reset()
            self.just_trained = False

        efference_copy = a + [self.simulation.fish.prev_action_impulse,
                              self.simulation.fish.prev_action_angle]

        o, r, internal_state, d, full_masked_image = self.simulation.simulation_step(action=a)

        self.buffer.action_buffer.append(efference_copy)  # Add to buffer for loading of previous actions


        while self.step_number < self.current_episode_max_duration:

            if self.assay is not None:
                # Deal with interventions
                if self.interruptions:
                    o, efference_copy, internal_state = self.perform_interruptions(o, efference_copy, internal_state)

                self.previous_action = a

            self.step_number += 1

            r, internal_state, o, d, rnn_state, rnn_state_ref, a = self.step_loop(
                o=o,
                internal_state=internal_state,
                a=a,
                rnn_state=rnn_state,
                rnn_state_ref=rnn_state_ref,
            )

            self.total_episode_reward += r
            if d:

                if self.run_version == "Original":
                    if self.simulation.switch_step != None:
                        self.buffer.switch_step = self.simulation.switch_step
                    else:
                        # If no split occurs, return without saving data.
                        print("No split occurred, as condition never met. Returning without saving data.")
                        return False

                break

        self.log_stimuli()

        if "reward assessments" in self.buffer.recordings:
            self.buffer.calculate_advantages_and_returns()

        if self.environment_params["salt"]:
            salt_location = self.simulation.salt_location
        else:
            salt_location = None

        if self.using_gpu:
            background = self.simulation.board.global_sediment_grating.get()[:, :, 0]
        else:
            background = self.simulation.board.global_sediment_grating[:, :, 0]

        self.buffer.save_assay_data(assay_id=assay['assay id'],
                                    data_save_location=self.data_save_location,
                                    assay_configuration_id=self.assay_configuration_id,
                                    internal_state_order=self.get_internal_state_order(),
                                    sediment=background,
                                    salt_location=salt_location)
        self.buffer.reset()
        if assay["save frames"]:
            episode_data = load_data(f"{self.model_name}-{self.model_number}", self.assay_configuration_id,
                                     assay['assay id'], training_data=False)
            draw_episode(episode_data, self.config_name, f"{self.model_name}-{self.model_number}",
                         self.continuous_actions,
                         save_id=f"{self.assay_configuration_id}-{assay['assay id']}")

        print(f"Assay: {assay['assay id']} Completed")
        print("")
        return True

    def step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref):
        return self._assay_step_loop(o, internal_state, a, rnn_state, rnn_state_ref)
