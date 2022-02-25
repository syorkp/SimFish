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

        # Configuration
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

        if switch_criteria_met:
            self.configuration_index = int(next_point)
            self.switched_configuration = True
            print(f"{self.model_id}: Changing configuration to configuration {self.configuration_index}")
            self.current_configuration_location = f"./Configurations/Training-Configs/{self.config_name}/{str(self.configuration_index)}"
            self.learning_params, self.environment_params = self.load_configuration_files()
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

        predators_avoided_summary = tf.Summary(
            value=[tf.Summary.Value(tag="predators avoided", simple_value=predators_avoided)])
        self.writer.add_summary(predators_avoided_summary, self.episode_number)

        sand_grains_bumped_summary = tf.Summary(
            value=[tf.Summary.Value(tag="attempted sand grain captures", simple_value=sand_grains_bumped)])
        self.writer.add_summary(sand_grains_bumped_summary, self.episode_number)

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


