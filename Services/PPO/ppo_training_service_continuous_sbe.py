from time import time
import json

import numpy as np
import tensorflow.compat.v1 as tf

from Analysis.Indexing.data_index_service import DataIndexServiceContinuous

from Buffers.PPO.ppo_buffer_continuous import PPOBufferContinuous
from Buffers.PPO.ppo_buffer_continuous_multivariate2 import PPOBufferContinuousMultivariate2

from Configurations.Templates.assay_config import naturalistic_assay_config
from Configurations.Utilities.turn_model_configs_into_assay_configs import transfer_config

from Services.PPO.continuous_ppo import ContinuousPPO
from Services.training_service import TrainingService
from Services.PPO.ppo_assay_service_continuous import ppo_assay_target_continuous

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_training_target_continuous_sbe(trial, total_steps, episode_number, memory_fraction, configuration_index):
    services = PPOTrainingServiceContinuousSBE(model_name=trial["Model Name"],
                                               trial_number=trial["Trial Number"],
                                               total_steps=total_steps,
                                               episode_number=episode_number,
                                               monitor_gpu=trial["monitor gpu"],
                                               using_gpu=trial["Using GPU"],
                                               memory_fraction=memory_fraction,
                                               config_name=trial["Environment Name"],
                                               realistic_bouts=trial["Realistic Bouts"],
                                               continuous_actions=trial["Continuous Actions"],
                                               new_simulation=trial["New Simulation"],

                                               model_exists=trial["Model Exists"],
                                               configuration_index=configuration_index,
                                               full_logs=trial["Full Logs"],
                                               profile_speed=trial["Profile Speed"],
                                               )
    print("Created service...", flush=True)
    services.run()


class PPOTrainingServiceContinuousSBE(TrainingService, ContinuousPPO):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, new_simulation, model_exists, configuration_index,
                 full_logs, profile_speed):
        super().__init__(model_name=model_name, trial_number=trial_number,
                         total_steps=total_steps, episode_number=episode_number,
                         monitor_gpu=monitor_gpu, using_gpu=using_gpu,
                         memory_fraction=memory_fraction, config_name=config_name,
                         realistic_bouts=realistic_bouts,
                         continuous_actions=continuous_actions,
                         new_simulation=new_simulation,
                         model_exists=model_exists,
                         configuration_index=configuration_index,
                         full_logs=full_logs,
                         profile_speed=profile_speed
                         )

        self.batch_size = self.learning_params["batch_size"]
        self.trace_length = self.learning_params["trace_length"]

        self.multivariate = self.learning_params["multivariate"]

        self.sb_emulator = True

        if self.multivariate:
            self.buffer = PPOBufferContinuousMultivariate2(gamma=self.learning_params["gamma"],
                                                           lmbda=self.learning_params["lambda"],
                                                           batch_size=self.learning_params["batch_size"],
                                                           train_length=self.learning_params["trace_length"],
                                                           assay=False,
                                                           debug=False,
                                                           use_dynamic_network=self.environment_params["use_dynamic_network"],
                                                           use_rnd=self.learning_params["use_rnd"]
                                                           )
        else:
            self.buffer = PPOBufferContinuous(gamma=self.learning_params["gamma"],
                                              lmbda=self.learning_params["lambda"],
                                              batch_size=self.learning_params["batch_size"],
                                              train_length=self.learning_params["trace_length"],
                                              assay=False,
                                              debug=False,
                                              use_dynamic_network=self.environment_params["use_dynamic_network"],
                                              )

        # IF not saving regular gifs, instead be ready to save the environmental data underlying GIFs.
        if not self.learning_params["save_gifs"]:
            if self.multivariate:
                self.episode_buffer = PPOBufferContinuousMultivariate2(gamma=self.learning_params["gamma"],
                                                                       lmbda=self.learning_params["lambda"],
                                                                       batch_size=self.learning_params["batch_size"],
                                                                       train_length=self.learning_params["trace_length"],
                                                                       assay=True,
                                                                       debug=False,
                                                                       use_dynamic_network=self.environment_params["use_dynamic_network"],
                                                                       )
            else:
                self.episode_buffer = PPOBufferContinuous(gamma=self.learning_params["gamma"],
                                                          lmbda=self.learning_params["lambda"],
                                                          batch_size=self.learning_params["batch_size"],
                                                          train_length=self.learning_params["trace_length"],
                                                          assay=True,
                                                          debug=False,
                                                          use_dynamic_network=self.environment_params[
                                                              "use_dynamic_network"],
                                                          )
        else:
            self.episode_buffer = False

        if self.learning_params["epsilon_greedy"]:
            self.epsilon_greedy = True
            self.e = self.learning_params["startE"]
        else:
            self.epsilon_greedy = False

        self.step_drop = (self.learning_params['startE'] - self.learning_params['endE']) / self.learning_params[
            'anneling_steps']

        self.use_rnd = self.learning_params["use_rnd"]

        # Whether or not to split the networks
        if "separate_networks" in self.learning_params:
            self.separate_networks = self.learning_params["separate_networks"]
        else:
            self.separate_networks = False

    def run(self):
        sess = self.create_session()
        print("Creating session...", flush=True)
        with sess as self.sess:
            self.create_network()
            self.init_states()
            TrainingService._run(self)
            self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")
            # Save the parameters to be carried over.
            output_data = {"epsilon": self.epsilon, "episode_number": self.episode_number,
                           "total_steps": self.total_steps, "configuration_index": self.configuration_index}
            with open(f"{self.model_location}/saved_parameters.json", "w") as file:
                json.dump(output_data, file)

        while self.switch_network_configuration:
            tf.reset_default_graph()
            sess = self.create_session()
            print("Switching network configuration...")
            with sess as self.sess:
                self.create_network()
                new_output_layer = self.actor_network.processing_network_output

                if new_output_layer != self.original_output_layer:  # If altered shape of final output layer
                    self.additional_layers += []  # TODO: Consider may need to remove final layers if changing processing_network_output.

                self.original_output_layer = None
                self.init_states()
                TrainingService._run(self)

        print("Training finished, starting to gather data...")
        tf.reset_default_graph()

        # Create Assay Config from Training Config
        transfer_config(self.model_name, self.model_name)

        # Build trial
        trial = naturalistic_assay_config
        trial["Model Name"] = self.model_name
        trial["Environment Name"] = self.model_name
        trial["Trial Number"] = self.model_number
        trial["Continuous Actions"] = True
        trial["Learning Algorithm"] = "PPO"
        for i, assay in enumerate(trial["Assays"]):
            trial["Assays"][i]["duration"] = self.learning_params["max_epLength"]
            trial["Assays"][i]["save frames"] = False

        # Run data gathering
        ppo_assay_target_continuous(trial, self.total_steps, self.episode_number, self.memory_fraction)
        # TODO: might need to clear data.

        # Perform cursory analysis on data
        data_index_service = DataIndexServiceContinuous(self.model_id)
        data_index_service.produce_behavioural_summary_display()

    def episode_loop(self):
        """
        Loops over an episode, which involves initialisation of the environment and RNN state, then iteration over the
        steps in the episode. The relevant values are then saved to the experience buffer.
        """
        t0 = time()

        self.current_episode_max_duration = self.learning_params["max_epLength"]
        self._episode_loop()

        # Train the network on the episode buffer
        self.buffer.calculate_advantages_and_returns()

        if self.use_rnd:
            self.buffer.update_rewards_rnd()

        if self.multivariate:
            if self.separate_networks:
                ContinuousPPO.train_network_multivariate2_split_networks(self)
            else:
                ContinuousPPO.train_network_multivariate2(self)

        else:
            ContinuousPPO.train_network(self)

        # Add the episode to tensorflow logs
        self.save_episode(episode_start_t=t0,
                          total_episode_reward=self.total_episode_reward,
                          prey_caught=self.simulation.prey_caught,
                          predators_avoided=self.simulation.predator_attacks_avoided,
                          sand_grains_bumped=self.simulation.sand_grains_bumped,
                          steps_near_vegetation=self.simulation.steps_near_vegetation,
                          )
        print(f"""{self.model_id} - episode {str(self.episode_number)}: num steps = {str(self.simulation.num_steps)}
Mean Impulse: {np.mean([i[0] for i in self.buffer.action_buffer])}
Mean Angle {np.mean([i[1] for i in self.buffer.action_buffer])}
Total episode reward: {self.total_episode_reward}\n""")

    def step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                  rnn_state_critic_ref):
        if self.multivariate:
            if self.learning_params["beta_distribution"]:
                if self.full_logs:
                    return self._step_loop_multivariate_beta_sbe_full_logs(o, internal_state, a, rnn_state_actor,
                                                                      rnn_state_actor_ref, rnn_state_critic,
                                                                      rnn_state_critic_ref)
                else:
                    return self._step_loop_multivariate_beta_sbe_reduced_logs(o, internal_state, a, rnn_state_actor,
                                                                         rnn_state_actor_ref, rnn_state_critic,
                                                                         rnn_state_critic_ref)
            else:
                if self.separate_networks:
                    if self.full_logs:
                        return self._step_loop_multivariate_sbe_sn_full_logs(o, internal_state, a, rnn_state_actor,
                                                                          rnn_state_actor_ref, rnn_state_critic,
                                                                          rnn_state_critic_ref)
                    else:
                        return self._step_loop_multivariate_sbe_sn_reduced_logs(o, internal_state, a, rnn_state_actor,
                                                                             rnn_state_actor_ref, rnn_state_critic,
                                                                             rnn_state_critic_ref)
                else:
                    if self.full_logs:
                        return self._step_loop_multivariate_sbe_full_logs(o, internal_state, a, rnn_state_actor,
                                                                          rnn_state_actor_ref, rnn_state_critic,
                                                                          rnn_state_critic_ref)
                    else:
                        return self._step_loop_multivariate_sbe_reduced_logs(o, internal_state, a, rnn_state_actor,
                                                                             rnn_state_actor_ref, rnn_state_critic,
                                                                             rnn_state_critic_ref)
        else:
            if self.full_logs:
                return self._step_loop_full_logs(o, internal_state, a, rnn_state_actor,
                                                          rnn_state_actor_ref, rnn_state_critic,
                                                          rnn_state_critic_ref)
            else:
                return self._step_loop_reduced_logs(o, internal_state, a, rnn_state_actor,
                                                             rnn_state_actor_ref, rnn_state_critic,
                                                             rnn_state_critic_ref)

    def save_episode(self, episode_start_t, total_episode_reward, prey_caught,
                     predators_avoided, sand_grains_bumped, steps_near_vegetation):
        """
        Saves the episode the the experience buffer. Also creates a gif if at interval.
        """
        TrainingService._save_episode(self, episode_start_t, total_episode_reward, prey_caught,
                                      predators_avoided, sand_grains_bumped, steps_near_vegetation)

        TrainingService._save_episode_continuous_variables(self)

        output_data = {"episode_number": self.episode_number, "total_steps": self.total_steps, "configuration_index": self.configuration_index}
        with open(f"{self.model_location}/saved_parameters.json", "w") as file:
            json.dump(output_data, file)

        # # Value Summary
        # for step in range(0, len(self.buffer.value_buffer)):
        #     value_summary = tf.Summary(
        #         value=[tf.Summary.Value(tag="value_predictions", simple_value=self.buffer.value_buffer[step])])
        #     self.writer.add_summary(value_summary, self.total_steps - len(self.buffer.value_buffer) + step)

