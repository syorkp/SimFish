import json
import os
import time
import numpy as np
import pstats

import tensorflow.compat.v1 as tf

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Services.base_service import BaseService
from Tools.make_gif import make_gif
from Tools.graph_functions import update_target_graph, update_target

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class TrainingService(BaseService):
    # TODO: Test new configuration savinfg
    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, new_simulation, model_exists, episode_transitions,
                 total_configurations, conditional_transitions, configuration_index, full_logs):

        super().__init__(model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                         config_name, realistic_bouts, continuous_actions, new_simulation)

        print("TrainingService Constructor called")

        # Configuration1
        self.total_configurations = total_configurations
        self.episode_transitions = episode_transitions
        self.conditional_transitions = conditional_transitions
        if configuration_index is not None:
            self.configuration_index = configuration_index
        else:
            self.configuration_index = 1
        self.switched_configuration = False
        self.current_configuration_location = f"./Configurations/Training-Configs/{self.config_name}/{str(self.configuration_index)}"
        self.learning_params, self.environment_params = self.load_configuration_files()

        # Training Parameters
        self.load_model = model_exists
        self.trainables = None
        self.target_ops = None
        self.writer = None
        self.algorithm = None

        # Create simulation variable
        self.create_environment()

        # Training Data
        self.training_times = []
        self.reward_list = []
        self.full_logs = full_logs

        # For config scaffolding

        self.previous_config_switch = self.episode_number

        self.last_episodes_prey_caught = []
        self.last_episodes_predators_avoided = []
        self.last_episodes_sand_grains_bumped = []

        # For debugging show mask
        self.visualise_mask = self.environment_params['visualise_mask']

        rnns = [layer for layer in self.learning_params["base_network_layers"].keys() if
                self.learning_params["base_network_layers"][layer][0] == "dynamic_rnn"] + \
               [layer for layer in self.learning_params["modular_network_layers"].keys() if
                self.learning_params["modular_network_layers"][layer][0] == "dynamic_rnn"]
        self.rnn_in_network = True if len(rnns) > 0 else False

    def _run(self):
        self.saver = tf.train.Saver(max_to_keep=5)
        self.init = tf.global_variables_initializer()
        self.trainables = tf.trainable_variables()
        if self.algorithm == "DQN":
            self.target_ops = update_target_graph(self.trainables, self.learning_params['tau'])

        if self.load_model:
            print(f"Attempting to load model at {self.model_location}")
            checkpoint = tf.train.get_checkpoint_state(self.model_location)
            if hasattr(checkpoint, "model_checkpoint_path"):
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("Loading successful")

            else:
                print("No saved checkpoints found, starting from scratch.")
                self.sess.run(self.init)
        else:
            print("First attempt at running model. Starting from scratch.")
            self.sess.run(self.init)
        self.writer = tf.summary.FileWriter(f"{self.model_location}/logs/", tf.get_default_graph())
        if self.algorithm == "DQN":
            update_target(self.target_ops, self.sess)  # Set the target network to be equal to the primary network.

        for e_number in range(self.episode_number, self.learning_params["num_episodes"]):
            self.episode_number = e_number
            if self.configuration_index < self.total_configurations:
                self.check_update_configuration()
            self.episode_loop()
            if self.monitor_performance:
                ps = pstats.Stats(self.profile)
                ps.sort_stats("tottime")
                ps.print_stats(20)
                print(time.time())

    def create_environment(self):
        if self.continuous_actions:
            self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts, self.new_simulation, self.using_gpu)
        else:
            self.simulation = DiscreteNaturalisticEnvironment(self.environment_params, self.realistic_bouts, self.new_simulation, self.using_gpu)

    def check_update_configuration(self):
        next_point = str(self.configuration_index + 1)
        episode_transition_points = self.episode_transitions.keys()
        switch_criteria_met = False

        # Switch config by episode
        if next_point in episode_transition_points and self.episode_number > self.episode_transitions[next_point]:
            switch_criteria_met = True
        elif len(self.last_episodes_prey_caught) >= 20:  # Switch config by behavioural conditionals

            prey_conditional_transition_points = self.conditional_transitions["Prey Caught"].keys()
            predators_conditional_transition_points = self.conditional_transitions["Predators Avoided"].keys()
            grains_bumped_conditional_transfer_points = self.conditional_transitions["Sand Grains Bumped"].keys()

            if next_point in predators_conditional_transition_points and \
                    np.mean(self.last_episodes_predators_avoided) > self.conditional_transitions["Predators Avoided"][next_point]:
                switch_criteria_met = True

            elif next_point in prey_conditional_transition_points and \
                    np.mean(self.last_episodes_prey_caught) > self.conditional_transitions["Prey Caught"][next_point]:
                switch_criteria_met = True

            elif next_point in grains_bumped_conditional_transfer_points and \
                    np.mean(self.last_episodes_sand_grains_bumped) > self.conditional_transitions["Sand Grains Bumped"][next_point]:
                switch_criteria_met = True

            if self.episode_number - self.previous_config_switch < 20:
                switch_criteria_met = False

        if switch_criteria_met:
            self.configuration_index = int(next_point)
            self.switched_configuration = True
            print(f"{self.model_id}: Changing configuration to configuration {self.configuration_index}")
            self.current_configuration_location = f"./Configurations/Training-Configs/{self.config_name}/{str(self.configuration_index)}"
            self.learning_params, self.environment_params = self.load_configuration_files()
            self.previous_config_switch = self.episode_number
            self.create_environment()
        else:
            self.switched_configuration = False

    def _save_episode(self, episode_start_t, total_episode_reward, prey_caught, predators_avoided, sand_grains_bumped,
                      steps_near_vegetation):
        """Saves episode data common to all models"""
        print(f"{self.model_id} - episode {str(self.episode_number)}: num steps = {str(self.simulation.num_steps)}",
              flush=True)

        # # Log the average training time for episodes (when not saved)
        # if not self.save_frames:
        #     self.training_times.append(time() - episode_start_t)
        #     print(np.mean(self.training_times))

        # Keep recent predators caught for configuration change conditionals
        self.last_episodes_prey_caught.append(prey_caught)
        self.last_episodes_predators_avoided.append(predators_avoided)
        self.last_episodes_sand_grains_bumped.append(sand_grains_bumped)
        if len(self.last_episodes_predators_avoided) > 20:
            self.last_episodes_prey_caught.pop(0)
            self.last_episodes_predators_avoided.pop(0)
            self.last_episodes_sand_grains_bumped.pop(0)

        # SUMMARIES
        # Rewards
        episode_summary = tf.Summary(value=[tf.Summary.Value(tag="episode reward", simple_value=total_episode_reward)])
        self.writer.add_summary(episode_summary, self.total_steps)

        #                  Environmental Logs                   #

        # Raw logs
        prey_caught_summary = tf.Summary(value=[tf.Summary.Value(tag="prey caught", simple_value=prey_caught)])
        self.writer.add_summary(prey_caught_summary, self.episode_number)

        if self.environment_params["probability_of_predator"] > 0:
            predators_avoided_summary = tf.Summary(
                value=[tf.Summary.Value(tag="predators avoided", simple_value=predators_avoided)])
            self.writer.add_summary(predators_avoided_summary, self.episode_number)

        if self.environment_params["sand_grain_num"] > 0:
            sand_grains_bumped_summary = tf.Summary(
                value=[tf.Summary.Value(tag="attempted sand grain captures", simple_value=sand_grains_bumped)])
            self.writer.add_summary(sand_grains_bumped_summary, self.episode_number)

        if self.environment_params["vegetation_num"] > 0:
            steps_near_vegetation_summary = tf.Summary(
                value=[tf.Summary.Value(tag="steps near vegetation", simple_value=steps_near_vegetation)])
            self.writer.add_summary(steps_near_vegetation_summary, self.episode_number)

        # Normalised Logs
        if self.environment_params["prey_num"] != 0:
            fraction_prey_caught = prey_caught / self.environment_params["prey_num"]
            prey_caught_summary = tf.Summary(
                value=[tf.Summary.Value(tag="prey capture index (fraction caught)", simple_value=fraction_prey_caught)])
            self.writer.add_summary(prey_caught_summary, self.episode_number)

        if self.environment_params["probability_of_predator"] != 0:
            predator_avoided_index = predators_avoided / self.environment_params["probability_of_predator"]
            predators_avoided_summary = tf.Summary(
                value=[tf.Summary.Value(tag="predator avoidance index (avoided/p_pred)",
                                        simple_value=predator_avoided_index)])
            self.writer.add_summary(predators_avoided_summary, self.episode_number)

        if self.environment_params["sand_grain_num"] != 0:
            sand_grain_capture_index = sand_grains_bumped / self.environment_params["sand_grain_num"]
            sand_grains_bumped_summary = tf.Summary(
                value=[tf.Summary.Value(tag="sand grain capture index (fraction attempted caught)",
                                        simple_value=sand_grain_capture_index)])
            self.writer.add_summary(sand_grains_bumped_summary, self.episode_number)

        if self.environment_params["vegetation_num"] != 0:
            vegetation_index = (steps_near_vegetation / self.simulation.num_steps) / self.environment_params[
                "vegetation_num"]
            use_of_vegetation_summary = tf.Summary(
                value=[tf.Summary.Value(tag="use of vegetation index (fraction_steps/vegetation_num",
                                        simple_value=vegetation_index)])
            self.writer.add_summary(use_of_vegetation_summary, self.episode_number)

        if self.switched_configuration:
            configuration_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Configuration change", simple_value=self.configuration_index)]
            )
            self.writer.add_summary(configuration_summary, self.episode_number)

        # Periodically save the model.
        if self.episode_number % self.learning_params['summaryLength'] == 0 and self.episode_number != 0:
            # print(f"mean time: {np.mean(self.training_times)}")

            # Save the model
            self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")
            print("Saved Model")

            # Create the GIF
            make_gif(self.frame_buffer, f"{self.model_location}/episodes/episode-{str(self.episode_number)}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
            if self.visualise_mask:
                make_gif(self.simulation.mask_buffer, f"{self.model_location}/episodes/mask-buffer-episode-{str(self.episode_number)}.gif",
                     duration=len(self.simulation.mask_buffer) * self.learning_params['time_per_step'], true_image=True)
            self.frame_buffer = []
            self.save_frames = False

        if (self.episode_number + 1) % self.learning_params['summaryLength'] == 0:
            print('starting to save frames', flush=True)
            self.save_frames = True
        if self.monitor_gpu:
            print(f"GPU usage {os.system('gpustat -cp')}")

        self.reward_list.append(total_episode_reward)

    def _save_episode_discrete_variables(self):
        values = self.buffer.value_buffer
        mean_value = np.mean(values)
        max_value = np.max(values)
        min_value = np.min(values)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="mean value", simple_value=mean_value)])
        self.writer.add_summary(value_summary, self.episode_number)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="max value", simple_value=max_value)])
        self.writer.add_summary(value_summary, self.episode_number)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="min value", simple_value=min_value)])
        self.writer.add_summary(value_summary, self.episode_number)

        if self.full_logs:
            # Save Loss
            mean_critic_loss = np.mean(self.buffer.critic_loss_buffer)
            max_critic_loss = np.max(self.buffer.critic_loss_buffer)
            min_critic_loss = np.min(self.buffer.critic_loss_buffer)
            critic_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean critic_loss", simple_value=mean_critic_loss)])
            self.writer.add_summary(critic_loss_summary, self.episode_number)
            critic_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="max critic_loss", simple_value=max_critic_loss)])
            self.writer.add_summary(critic_loss_summary, self.episode_number)
            critic_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="min critic_loss", simple_value=min_critic_loss)])
            self.writer.add_summary(critic_loss_summary, self.episode_number)

            mean_actor_loss = np.mean(self.buffer.actor_loss_buffer)
            max_actor_loss = np.max(self.buffer.actor_loss_buffer)
            min_actor_loss = np.min(self.buffer.actor_loss_buffer)
            actor_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean actor_loss", simple_value=mean_actor_loss)])
            self.writer.add_summary(actor_loss_summary, self.episode_number)
            actor_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="max actor_loss", simple_value=max_actor_loss)])
            self.writer.add_summary(actor_loss_summary, self.episode_number)
            actor_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="min actor_loss", simple_value=min_actor_loss)])
            self.writer.add_summary(actor_loss_summary, self.episode_number)

    def _save_episode_continuous_variables(self):
        impulses = self.buffer.action_buffer[:, 0]
        mean_impulse = np.mean(impulses)
        max_impulse = np.max(impulses)
        min_impulse = np.min(impulses)
        impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="mean impulse", simple_value=mean_impulse)])
        self.writer.add_summary(impulse_summary, self.episode_number)
        impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="max impulse", simple_value=max_impulse)])
        self.writer.add_summary(impulse_summary, self.episode_number)
        impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="min impulse", simple_value=min_impulse)])
        self.writer.add_summary(impulse_summary, self.episode_number)

        angles = self.buffer.action_buffer[:, 1]
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
        values = self.buffer.value_buffer
        mean_value = np.mean(values)
        max_value = np.max(values)
        min_value = np.min(values)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="mean value", simple_value=mean_value)])
        self.writer.add_summary(value_summary, self.episode_number)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="max value", simple_value=max_value)])
        self.writer.add_summary(value_summary, self.episode_number)
        value_summary = tf.Summary(value=[tf.Summary.Value(tag="min value", simple_value=min_value)])
        self.writer.add_summary(value_summary, self.episode_number)

        if self.full_logs:
            # Save Loss
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

        #                OLD

        # Action Summary
        # impulses = [action[0] for action in self.buffer.action_buffer]
        # for step in range(0, len(impulses), 5):
        #     impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="impulse magnitude", simple_value=impulses[step])])
        #     self.writer.add_summary(impulse_summary, self.total_steps - len(impulses) + step)
        #
        # angles = [action[1] for action in self.buffer.action_buffer]
        # for step in range(0, len(angles), 5):
        #     angles_summary = tf.Summary(value=[tf.Summary.Value(tag="angle magnitude", simple_value=angles[step])])
        #     self.writer.add_summary(angles_summary, self.total_steps - len(angles) + step)

        # if self.full_logs:
        #     # Save Loss
        #     for step in range(0, len(self.buffer.critic_loss_buffer)):
        #         critic_loss_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="critic loss", simple_value=self.buffer.critic_loss_buffer[step])])
        #         self.writer.add_summary(critic_loss_summary,
        #                                 self.total_steps - len(angles) + step * self.learning_params["batch_size"])
        #
        #     for step in range(0, len(self.buffer.impulse_loss_buffer)):
        #         impulse_loss_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="impulse loss", simple_value=self.buffer.impulse_loss_buffer[step])])
        #         self.writer.add_summary(impulse_loss_summary,
        #                                 self.total_steps - len(angles) + step * self.learning_params["batch_size"])
        #
        #     for step in range(0, len(self.buffer.angle_loss_buffer)):
        #         angle_loss_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="angle loss", simple_value=self.buffer.angle_loss_buffer[step])])
        #         self.writer.add_summary(angle_loss_summary,
        #                                 self.total_steps - len(angles) + step * self.learning_params["batch_size"])
        #
        #     # Saving Parameters for Testing
        #     for step in range(0, len(self.buffer.mu_i_buffer)):
        #         mu_i_loss_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="mu_impulse", simple_value=self.buffer.mu_i_buffer[step])])
        #         self.writer.add_summary(mu_i_loss_summary, self.total_steps - len(self.buffer.mu_i_buffer) + step)
        #
        #     for step in range(0, len(self.buffer.si_i_buffer)):
        #         si_i_loss_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="sigma_impulse", simple_value=self.buffer.si_i_buffer[step])])
        #         self.writer.add_summary(si_i_loss_summary, self.total_steps - len(self.buffer.si_i_buffer) + step)
        #
        #     for step in range(0, len(self.buffer.mu_a_buffer)):
        #         mu_a_loss_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="mu_angle", simple_value=self.buffer.mu_a_buffer[step])])
        #         self.writer.add_summary(mu_a_loss_summary, self.total_steps - len(self.buffer.mu_a_buffer) + step)
        #
        #     for step in range(0, len(self.buffer.si_a_buffer)):
        #         si_a_loss_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="sigma_angle", simple_value=self.buffer.si_a_buffer[step])])
        #         self.writer.add_summary(si_a_loss_summary, self.total_steps - len(self.buffer.si_a_buffer) + step)
        #
        #     for step in range(0, len(self.buffer.mu1_buffer)):
        #         mu1_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="mu_impulse_base", simple_value=self.buffer.mu1_buffer[step])])
        #         self.writer.add_summary(mu1_summary, self.total_steps - len(self.buffer.mu1_buffer) + step)
        #
        #     for step in range(0, len(self.buffer.mu1_ref_buffer)):
        #         mu1_ref_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="mu_impulse_ref_base", simple_value=self.buffer.mu1_ref_buffer[step])])
        #         self.writer.add_summary(mu1_ref_summary, self.total_steps - len(self.buffer.mu1_ref_buffer) + step)
        #
        #     for step in range(0, len(self.buffer.mu_a1_buffer)):
        #         mu1_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="mu_angle_base", simple_value=self.buffer.mu_a1_buffer[step])])
        #         self.writer.add_summary(mu1_summary, self.total_steps - len(self.buffer.mu_a1_buffer) + step)
        #
        #     for step in range(0, len(self.buffer.mu_a_ref_buffer)):
        #         mu1_ref_summary = tf.Summary(
        #             value=[tf.Summary.Value(tag="mu_angle_ref_base", simple_value=self.buffer.mu_a_ref_buffer[step])])
        #         self.writer.add_summary(mu1_ref_summary, self.total_steps - len(self.buffer.mu_a_ref_buffer) + step)
