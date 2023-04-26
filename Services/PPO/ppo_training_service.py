import json

import numpy as np

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Analysis.Indexing.data_index_service import DataIndexServiceContinuous
from Analysis.Behavioural.Exploration.turn_chain_metric import get_normalised_turn_chain_metric_continuous

from Buffers.PPO.ppo_buffer import PPOBuffer

from Configurations.Utilities.turn_model_configs_into_assay_configs import transfer_config

from Services.PPO.continuous_ppo import ContinuousPPO
from Services.training_service import TrainingService
from Services.PPO.ppo_assay_service import ppo_assay_target_continuous

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_training_target_continuous_sbe(trial, epsilon, total_steps, episode_number, memory_fraction, configuration_index):
    if "Using GPU" in trial:
        using_gpu = trial["Using GPU"]
    else:
        using_gpu = True

    if "Profile Speed" in trial:
        profile_speed = trial["Profile Speed"]
    else:
        profile_speed = False

    services = PPOTrainingService(model_name=trial["Model Name"],
                                  trial_number=trial["Trial Number"],
                                  total_steps=total_steps,
                                  episode_number=episode_number,
                                  using_gpu=using_gpu,
                                  memory_fraction=memory_fraction,
                                  config_name=trial["Environment Name"],
                                  model_exists=trial["Model Exists"],
                                  epsilon=epsilon,
                                  configuration_index=configuration_index,
                                  profile_speed=profile_speed,
                                  )
    print("Created service...", flush=True)
    services.run()


class PPOTrainingService(TrainingService, ContinuousPPO):

    def __init__(self, model_name, trial_number, total_steps, episode_number, using_gpu, memory_fraction,
                 config_name, model_exists, epsilon, configuration_index, profile_speed):
        super().__init__(model_name=model_name,
                         trial_number=trial_number,
                         total_steps=total_steps,
                         episode_number=episode_number,
                         using_gpu=using_gpu,
                         memory_fraction=memory_fraction,
                         config_name=config_name,
                         continuous_actions=True,
                         model_exists=model_exists,
                         configuration_index=configuration_index,
                         profile_speed=profile_speed
                         )

        self.algorithm = "PPO"

        self.batch_size = self.learning_params["batch_size"]
        self.trace_length = self.learning_params["trace_length"]

        self.sb_emulator = True

        self.buffer = PPOBuffer(gamma=self.learning_params["gamma"],
                                lmbda=self.learning_params["lambda"],
                                batch_size=self.learning_params["batch_size"],
                                train_length=self.learning_params["trace_length"],
                                assay=False,
                                debug=False,
                                )

        # Save data from episode for video creation.
        self.episode_buffer = PPOBuffer(gamma=self.learning_params["gamma"],
                                        lmbda=self.learning_params["lambda"],
                                        batch_size=self.learning_params["batch_size"],
                                        train_length=self.learning_params["trace_length"],
                                        assay=True,
                                        debug=False,
                                        )

        if self.learning_params["epsilon_greedy"]:
            self.epsilon_greedy = True
            if epsilon is None:
                self.epsilon = self.learning_params["startE"]
            else:
                self.epsilon = epsilon
        else:
            self.epsilon_greedy = False

        self.step_drop = (self.learning_params['startE'] - self.learning_params['endE']) / self.learning_params[
            'anneling_steps']

        self.last_position_dim = self.environment_params["prey_num"]

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

        print("Training finished")

    def episode_loop(self):
        """
        Loops over an episode, which involves initialisation of the environment and RNN state, then iteration over the
        steps in the episode. The relevant values are then saved to the experience buffer.
        """
        self.current_episode_max_duration = self.learning_params["max_epLength"]
        self._episode_loop()

        # Train the network on the episode buffer
        self.buffer.calculate_advantages_and_returns()

        ContinuousPPO.train_network(self)

        # Add the episode to tensorflow logs
        self.save_episode(total_episode_reward=self.total_episode_reward,
                          prey_caught=self.simulation.prey_caught,
                          sand_grains_bumped=self.simulation.sand_grains_bumped,
                          )
        print(f"""{self.model_id} - episode {str(self.episode_number)}: num steps = {str(self.simulation.num_steps)}
Mean Impulse: {np.mean([i[0] for i in self.buffer.action_buffer])}
Mean Angle {np.mean([i[1] for i in self.buffer.action_buffer])}
Total episode reward: {self.total_episode_reward}\n""", flush=True)

    def save_episode(self, total_episode_reward, prey_caught, sand_grains_bumped):
        """
        Saves the episode the experience buffer. Also creates a gif if at interval.
        """
        self._save_episode_continuous_variables()
        TrainingService._save_episode(self, total_episode_reward, prey_caught, sand_grains_bumped)

        output_data = {"episode_number": self.episode_number,
                       "total_steps": self.total_steps,
                       "configuration_index": self.configuration_index}
        with open(f"{self.model_location}/saved_parameters.json", "w") as file:
            json.dump(output_data, file)

    def _save_episode_continuous_variables(self):
        """Adds variables unique to PPO to tensorflow logs."""

        impulses = np.array(self.buffer.action_buffer)[:, 0]
        mean_impulse = np.mean(impulses)
        max_impulse = np.max(impulses)
        min_impulse = np.min(impulses)
        impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="mean impulse", simple_value=mean_impulse)])
        self.writer.add_summary(impulse_summary, self.episode_number)
        impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="max impulse", simple_value=max_impulse)])
        self.writer.add_summary(impulse_summary, self.episode_number)
        impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="min impulse", simple_value=min_impulse)])
        self.writer.add_summary(impulse_summary, self.episode_number)

        angles = np.array(self.buffer.action_buffer)[:, 1]
        mean_angle = np.mean(angles)
        max_angle = np.max(angles)
        min_angle = np.min(angles)
        angle_summary = tf.Summary(value=[tf.Summary.Value(tag="mean angle", simple_value=mean_angle)])
        self.writer.add_summary(angle_summary, self.episode_number)
        angle_summary = tf.Summary(value=[tf.Summary.Value(tag="max angle", simple_value=max_angle)])
        self.writer.add_summary(angle_summary, self.episode_number)
        angle_summary = tf.Summary(value=[tf.Summary.Value(tag="min angle", simple_value=min_angle)])
        self.writer.add_summary(angle_summary, self.episode_number)

        # # Value Summary
        values = np.array(self.buffer.value_buffer)
        mean_value = np.mean(values)
        max_value = np.max(values)
        min_value = np.min(values)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="mean value", simple_value=mean_value)])
        self.writer.add_summary(value_summary, self.episode_number)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="max value", simple_value=max_value)])
        self.writer.add_summary(value_summary, self.episode_number)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="min value", simple_value=min_value)])
        self.writer.add_summary(value_summary, self.episode_number)

        # Turn Chain metric
        turn_chain_preference = get_normalised_turn_chain_metric_continuous(angles)
        turn_chain_preference_summary = tf.Summary(value=[tf.Summary.Value(tag="turn chain preference", simple_value=turn_chain_preference)])
        self.writer.add_summary(turn_chain_preference_summary, self.episode_number)

        # Action Diversity
        impulse_choice_range = np.max(np.absolute(self.buffer.mu_i_buffer)) - np.min(np.absolute(self.buffer.mu_i_buffer))
        angle_choice_range = np.max(np.absolute(self.buffer.mu_a_buffer)) - np.min(np.absolute(self.buffer.mu_a_buffer))
        impulse_choice_range_prop = impulse_choice_range / self.environment_params["max_impulse"]
        angle_choice_range_prop = angle_choice_range / self.environment_params["max_angle_change"]

        impulse_action_diversity = tf.Summary(value=[tf.Summary.Value(tag="Impulse Choice Diversity", simple_value=impulse_choice_range_prop)])
        self.writer.add_summary(impulse_action_diversity, self.episode_number)

        angle_action_diversity = tf.Summary(value=[tf.Summary.Value(tag="Angle Choice Diversity", simple_value=angle_choice_range_prop)])
        self.writer.add_summary(angle_action_diversity, self.episode_number)

        if len(self.buffer.critic_loss_buffer) > 0:
            mean_critic_loss = np.mean(self.buffer.critic_loss_buffer)
            max_critic_loss = np.max(self.buffer.critic_loss_buffer)
            min_critic_loss = np.min(self.buffer.critic_loss_buffer)
            critic_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean critic_loss", simple_value=mean_critic_loss)])
            self.writer.add_summary(critic_loss_summary, self.episode_number)
            critic_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="max critic_loss", simple_value=max_critic_loss)])
            self.writer.add_summary(critic_loss_summary, self.episode_number)
            critic_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="min critic_loss", simple_value=min_critic_loss)])
            self.writer.add_summary(critic_loss_summary, self.episode_number)

            mean_impulse_loss = np.mean(self.buffer.impulse_loss_buffer)
            max_impulse_loss = np.max(self.buffer.impulse_loss_buffer)
            min_impulse_loss = np.min(self.buffer.impulse_loss_buffer)
            impulse_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean impulse_loss", simple_value=mean_impulse_loss)])
            self.writer.add_summary(impulse_loss_summary, self.episode_number)
            impulse_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="max impulse_loss", simple_value=max_impulse_loss)])
            self.writer.add_summary(impulse_loss_summary, self.episode_number)
            impulse_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="min impulse_loss", simple_value=min_impulse_loss)])
            self.writer.add_summary(impulse_loss_summary, self.episode_number)

            mean_angle_loss = np.mean(self.buffer.angle_loss_buffer)
            max_angle_loss = np.max(self.buffer.angle_loss_buffer)
            min_angle_loss = np.min(self.buffer.angle_loss_buffer)
            angle_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean angle_loss", simple_value=mean_angle_loss)])
            self.writer.add_summary(angle_loss_summary, self.episode_number)
            angle_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="max angle_loss", simple_value=max_angle_loss)])
            self.writer.add_summary(angle_loss_summary, self.episode_number)
            angle_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="min angle_loss", simple_value=min_angle_loss)])
            self.writer.add_summary(angle_loss_summary, self.episode_number)

            mean_entropy_loss = np.mean(self.buffer.entropy_loss_buffer)
            max_entropy_loss = np.max(self.buffer.entropy_loss_buffer)
            min_entropy_loss = np.min(self.buffer.entropy_loss_buffer)
            entropy_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean entropy_loss", simple_value=mean_entropy_loss)])
            self.writer.add_summary(entropy_loss_summary, self.episode_number)
            entropy_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="max entropy_loss", simple_value=max_entropy_loss)])
            self.writer.add_summary(entropy_loss_summary, self.episode_number)
            entropy_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="min entropy_loss", simple_value=min_entropy_loss)])
            self.writer.add_summary(entropy_loss_summary, self.episode_number)

        # Saving Parameters for Testing
        mean_mu_i = np.mean(self.buffer.mu_i_buffer)
        max_mu_i = np.max(self.buffer.mu_i_buffer)
        min_mu_i = np.min(self.buffer.mu_i_buffer)
        mu_i = tf.Summary(value=[tf.Summary.Value(tag="mean mu_i", simple_value=mean_mu_i)])
        self.writer.add_summary(mu_i, self.episode_number)
        mu_i = tf.Summary(value=[tf.Summary.Value(tag="max mu_i", simple_value=max_mu_i)])
        self.writer.add_summary(mu_i, self.episode_number)
        mu_i = tf.Summary(value=[tf.Summary.Value(tag="min mu_i", simple_value=min_mu_i)])
        self.writer.add_summary(mu_i, self.episode_number)

        mean_si_i = np.mean(self.buffer.si_i_buffer)
        max_si_i = np.max(self.buffer.si_i_buffer)
        min_si_i = np.min(self.buffer.si_i_buffer)
        si_i = tf.Summary(value=[tf.Summary.Value(tag="mean si_i", simple_value=mean_si_i)])
        self.writer.add_summary(si_i, self.episode_number)
        si_i = tf.Summary(value=[tf.Summary.Value(tag="max si_i", simple_value=max_si_i)])
        self.writer.add_summary(si_i, self.episode_number)
        si_i = tf.Summary(value=[tf.Summary.Value(tag="min si_i", simple_value=min_si_i)])
        self.writer.add_summary(si_i, self.episode_number)

        mean_mu_a = np.mean(self.buffer.mu_a_buffer)
        max_mu_a = np.max(self.buffer.mu_a_buffer)
        min_mu_a = np.min(self.buffer.mu_a_buffer)
        mu_a = tf.Summary(value=[tf.Summary.Value(tag="mean mu_a", simple_value=mean_mu_a)])
        self.writer.add_summary(mu_a, self.episode_number)
        mu_a = tf.Summary(value=[tf.Summary.Value(tag="max mu_a", simple_value=max_mu_a)])
        self.writer.add_summary(mu_a, self.episode_number)
        mu_a = tf.Summary(value=[tf.Summary.Value(tag="min mu_a", simple_value=min_mu_a)])
        self.writer.add_summary(mu_a, self.episode_number)

        mean_si_a = np.mean(self.buffer.si_a_buffer)
        max_si_a = np.max(self.buffer.si_a_buffer)
        min_si_a = np.min(self.buffer.si_a_buffer)
        si_a = tf.Summary(value=[tf.Summary.Value(tag="mean si_a", simple_value=mean_si_a)])
        self.writer.add_summary(si_a, self.episode_number)
        si_a = tf.Summary(value=[tf.Summary.Value(tag="max si_a", simple_value=max_si_a)])
        self.writer.add_summary(si_a, self.episode_number)
        si_a = tf.Summary(value=[tf.Summary.Value(tag="min si_a", simple_value=min_si_a)])
        self.writer.add_summary(si_a, self.episode_number)
