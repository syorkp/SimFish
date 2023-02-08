import numpy as np
import tensorflow.compat.v1 as tf
import copy

from Buffers.PPO.ppo_buffer_continuous import PPOBufferContinuous
from Buffers.PPO.ppo_buffer_continuous_multivariate import PPOBufferContinuousMultivariate

from Services.PPO.continuous_ppo import ContinuousPPO
from Services.assay_service import AssayService
from Analysis.Video.behaviour_video_construction import draw_episode
from Analysis.load_data import load_data

tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_assay_target_continuous(trial, total_steps, episode_number, memory_fraction):
    if "monitor gpu" in trial:
        monitor_gpu = trial["monitor gpu"]
    else:
        monitor_gpu = False

    if "Using GPU" in trial:
        using_gpu = trial["Using GPU"]
    else:
        using_gpu = True

    if "Continuous Actions" in trial:
        continuous_actions = trial["Continuous Actions"]
    else:
        continuous_actions = True

    if "Realistic Bouts" in trial:
        realistic_bouts = trial["Realistic Bouts"]
    else:
        realistic_bouts = True

    if "SB Emulator" in trial:
        sb_emulator = trial["SB Emulator"]
    else:
        sb_emulator = True

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
        if "Run Index" not in trial:
            run_version = "Original"
        else:
            modification = trial["Modification"]
            split_event = trial["Split Event"]
            run_version = trial["Run Index"]
    else:
        run_version = None

    service = PPOAssayServiceContinuous(model_name=trial["Model Name"],
                                        trial_number=trial["Trial Number"],
                                        total_steps=total_steps,
                                        episode_number=episode_number,
                                        monitor_gpu=monitor_gpu,
                                        using_gpu=using_gpu,
                                        memory_fraction=memory_fraction,
                                        config_name=trial["Environment Name"],
                                        realistic_bouts=realistic_bouts,
                                        continuous_environment=continuous_actions,
                                        assays=trial["Assays"],
                                        set_random_seed=set_random_seed,
                                        assay_config_name=trial["Assay Configuration Name"],

                                        sb_emulator=sb_emulator,
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

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_environment, assays, set_random_seed,
                 assay_config_name, sb_emulator, checkpoint, behavioural_recordings, network_recordings, interventions,
                 run_version, split_event, modification):
        """
        Runs a set of assays provided by the run configuraiton.
        """
        # Set random seed
        super().__init__(model_name=model_name,
                         trial_number=trial_number,
                         total_steps=total_steps,
                         episode_number=episode_number,
                         monitor_gpu=monitor_gpu,
                         using_gpu=using_gpu,
                         memory_fraction=memory_fraction,
                         config_name=config_name,
                         realistic_bouts=realistic_bouts,
                         continuous_environment=continuous_environment,
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

        self.multivariate = self.learning_params["multivariate"]

        # Buffer for saving results of assay
        if self.multivariate:
            self.buffer = PPOBufferContinuousMultivariate(gamma=0.99,
                                                          lmbda=0.9,
                                                          batch_size=self.learning_params["batch_size"],
                                                          train_length=self.learning_params["trace_length"],
                                                          assay=True,
                                                          debug=False,
                                                          use_dynamic_network=self.environment_params[
                                                              "use_dynamic_network"],
                                                          )
        else:
            self.buffer = PPOBufferContinuous(gamma=0.99,
                                              lmbda=0.9,
                                              batch_size=self.learning_params["batch_size"],
                                              train_length=self.learning_params["trace_length"],
                                              assay=True,
                                              debug=False,
                                              use_dynamic_network=self.environment_params["use_dynamic_network"],
                                              )

        self.ppo_version = ContinuousPPO
        self.use_rnd = self.learning_params["use_rnd"]
        self.sb_emulator = sb_emulator
        self.e = self.learning_params["startE"] / 2
        self.epsilon_greedy = self.learning_params["epsilon_greedy"]
        self.last_position_dim = self.environment_params["prey_num"]

    def run(self):
        sess = self.create_session()
        with sess as self.sess:
            self.create_network()
            self.init_states()
            AssayService._run(self)

    def perform_assay(self, assay, background=None, energy_state=None):
        # self.assay_output_data_format = {key: None for key in
        #                                  assay["behavioural recordings"] + assay["network recordings"]}
        # self.buffer.init_assay_recordings(assay["behavioural recordings"], assay["network recordings"])

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

        # TODO: implement environment loading etc as in dqn_assay_service. Will require modification of
        #  self._episode_loop() for RNN state, env reset, step num,

        if self.run_version == "Original-Completion" or self.run_version == "Modified-Completion":
            print("Loading Simulation")
            o = self.simulation.load_simulation(self.buffer, background, energy_state)
            internal_state = self.buffer.internal_state_buffer[-1]
            a = self.buffer.action_buffer[-1]

            a = np.array(a + [self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle])

            if self.run_version == "Modified-Completion":
                self.simulation.make_modification()

            a, updated_rnn_state, rnn2_state, network_layers, sa, sv = \
                self.sess.run(
                    [self.main_QN.predict, self.main_QN.rnn_state_shared, self.main_QN.rnn_state_ref,
                     self.main_QN.network_graph,

                     self.main_QN.streamA,
                     self.main_QN.streamV,
                     ],
                    feed_dict={self.main_QN.observation: o,
                               self.main_QN.internal_state: internal_state,
                               self.main_QN.prev_actions: np.expand_dims(a, 0),
                               self.main_QN.train_length: 1,
                               self.main_QN.rnn_state_in: rnn_state,
                               self.main_QN.rnn_state_in_ref: rnn_state_ref,

                               self.main_QN.batch_size: 1,
                               self.main_QN.exp_keep: 1.0,
                               # self.main_QN.learning_rate: self.learning_params["learning_rate"],
                               })

            self.step_number = len(self.buffer.internal_state_buffer)
        else:
            self.simulation.reset()
            a = [4.0, 0.0]
            self.step_number = 0

        if self.assay or self.just_trained:
            self.buffer.reset()
            self.just_trained = False

        action = a + [self.simulation.fish.prev_action_impulse,
                      self.simulation.fish.prev_action_angle]

        sa = np.zeros((1, 128))
        o, r, internal_state, d, FOV = self.simulation.simulation_step(action=a, activations=(sa,))

        self.buffer.action_buffer.append(action)  # Add to buffer for loading of previous actions

        self.step_number = 0
        while self.step_number < self.current_episode_max_duration:
            # print(self.step_number)
            if self.assay is not None:
                # Deal with interventions
                if self.visual_interruptions is not None:
                    if self.visual_interruptions[self.step_number] == 1:
                        # mean values over all data
                        o[:, 0, :] = 4
                        o[:, 1, :] = 11
                        o[:, 2, :] = 16
                if self.reafference_interruptions is not None:
                    if self.reafference_interruptions[self.step_number] is not False:
                        a = [self.reafference_interruptions[self.step_number]]
                if self.preset_energy_state is not None:
                    if self.preset_energy_state[self.step_number] is not False:
                        self.simulation.fish.energy_level = self.preset_energy_state[self.step_number]
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("energy_state")
                        internal_state[0, index] = self.preset_energy_state[self.step_number]
                if self.in_light_interruptions is not False:
                    if self.in_light_interruptions[self.step_number] == 1:
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("in_light")
                        internal_state[0, index] = self.in_light_interruptions[self.step_number]
                if self.salt_interruptions is not False:
                    if self.salt_interruptions[self.step_number] == 1:
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("salt")
                        internal_state[0, index] = self.salt_interruptions[self.step_number]

                self.previous_action = a

            self.step_number += 1

            r, internal_state, o, d, rnn_state, rnn_state_ref, _, __, a = self.step_loop(
                o=o,
                internal_state=internal_state,
                a=a,
                rnn_state_actor=rnn_state,
                rnn_state_actor_ref=rnn_state_ref,
                rnn_state_critic=rnn_state,
                rnn_state_critic_ref=rnn_state_ref
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


        #     # make_gif(self.frame_buffer,
        #     #          f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.gif",
        #     #          duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        #     make_video(self.frame_buffer,
        #              f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.mp4",
        #              duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        # self.frame_buffer = []

        if "reward assessments" in self.buffer.recordings:
            self.buffer.calculate_advantages_and_returns()

        if self.environment_params["salt"]:
            salt_location = self.simulation.salt_location
        else:
            salt_location = None

        if self.using_gpu:
            background = self.simulation.board.global_background_grating.get()[:, :, 0]
        else:
            background = self.simulation.board.global_background_grating[:, :, 0]

        self.buffer.save_assay_data(assay_id=assay['assay id'],
                                    data_save_location=self.data_save_location,
                                    assay_configuration_id=self.assay_configuration_id,
                                    internal_state_order=self.get_internal_state_order(),
                                    background=background,
                                    salt_location=salt_location)
        self.buffer.reset()
        if assay["save frames"]:
            episode_data = load_data(f"{self.model_name}-{self.model_number}", self.assay_configuration_id,
                                     assay['assay id'], training_data=False)
            draw_episode(episode_data, self.config_name, f"{self.model_name}-{self.model_number}", self.continuous_actions,
                         save_id=f"{self.assay_configuration_id}-{assay['assay id']}", training_episode=False)

        print(f"Assay: {assay['assay id']} Completed")
        print("")
        return True


    def step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                  rnn_state_critic_ref):
        if self.multivariate:
            if self.sb_emulator:
                return self._assay_step_loop_multivariate2(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                                           rnn_state_critic,
                                                           rnn_state_critic_ref)
            else:
                return self._assay_step_loop_multivariate(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                                          rnn_state_critic,
                                                          rnn_state_critic_ref)
        else:
            return self._assay_step_loop(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                                         rnn_state_critic_ref)
