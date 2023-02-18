import copy
import json
import os
import re
import time
import numpy as np
import pstats
import cProfile

import tensorflow.compat.v1 as tf

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Services.base_service import BaseService
from Tools.make_gif import make_gif
from Tools.make_video import make_video
from Tools.graph_functions import update_target_graph, update_target
from Analysis.Behavioural.Exploration.turn_chain_metric import get_normalised_turn_chain_metric_continuous
from Analysis.Video.behaviour_video_construction import draw_episode
from Analysis.load_data import load_data

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
# print(device_lib.list_local_devices())

class TrainingService(BaseService):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, model_exists, configuration_index,
                 full_logs, profile_speed):

        super().__init__(model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                         config_name, realistic_bouts, continuous_actions, profile_speed)

        print("TrainingService Constructor called")

        # Configurations
        self.total_configurations = None
        self.episode_transitions = None
        self.pci_transitions = None
        self.pai_transitions = None
        self.sgb_transitions = None
        self.finished_conditions = None
        self.load_transitions()
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
        self.switch_network_configuration = False
        self.additional_layers = None
        self.removed_layers = None
        self.original_output_layer = None

        self.last_episodes_prey_caught = []
        self.last_episodes_reward = []
        self.last_episodes_predators_avoided = []
        self.last_episodes_sand_grains_bumped = []

        # For debugging show mask
        self.visualise_mask = self.environment_params['visualise_mask']

        # Determining whether RNN exists in network
        rnns = [layer for layer in self.learning_params["base_network_layers"].keys() if
                self.learning_params["base_network_layers"][layer][0] == "dynamic_rnn"] + \
               [layer for layer in self.learning_params["modular_network_layers"].keys() if
                self.learning_params["modular_network_layers"][layer][0] == "dynamic_rnn"]
        self.rnn_in_network = True if len(rnns) > 0 else False

        # For regular saving
        self.save_environmental_data = False

        if "min_scaffold_interval" in self.learning_params:
            self.min_scaffold_interval = self.learning_params["min_scaffold_interval"]
        else:
            self.min_scaffold_interval = 20

    def _run(self):
        if self.switch_network_configuration:
            variables_to_keep = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            print("To remove:")
            print(self.additional_layers)
            variables_to_keep = self.remove_new_variables(variables_to_keep, self.additional_layers)
            self.saver = tf.train.Saver(max_to_keep=None, var_list=variables_to_keep)
        else:
            self.saver = tf.train.Saver(max_to_keep=None)

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
                checkpoint_path = checkpoint.model_checkpoint_path
                checkpoint_num = int(checkpoint_path.split("/model-")[-1][:-5])
                self.episode_number = checkpoint_num
                self.checkpoint_steps = self.total_steps

            else:
                print("No saved checkpoints found, starting from scratch.")
                self.sess.run(self.init)
                self.episode_number = 0
                self.checkpoint_steps = 0
        else:
            print("First attempt at running model. Starting from scratch.")
            self.sess.run(self.init)
            self.episode_number = 0
            self.checkpoint_steps = 0

        self.writer = tf.summary.FileWriter(f"{self.model_location}/logs/", tf.get_default_graph())

        if self.switch_network_configuration:
            # Re-initialise...
            self.sess.run(self.init)

            # Save values, to prevent an error.
            self.saver = tf.train.Saver(max_to_keep=None)
            self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")
            self.switch_network_configuration = False

        if self.algorithm == "DQN":
            update_target(self.target_ops, self.sess)  # Set the target network to be equal to the primary network.

        for e_number in range(self.episode_number, self.learning_params["num_episodes"]):
            self.episode_number = e_number
            if self.configuration_index < self.total_configurations:
                self.check_update_configuration()
            elif self.configuration_index == self.total_configurations:
                self.switched_configuration = False
                print("Reached final config...")
                if len(self.last_episodes_prey_caught) >= self.min_scaffold_interval:
                    # if np.mean(self.last_episodes_predators_avoided) / self.environment_params["probability_of_predator"] \
                    #         > self.finished_conditions["PAI"] \
                    #         and np.mean(self.last_episodes_prey_caught)/self.environment_params["prey_num"] \
                    #         > self.finished_conditions["PCI"]:

                    if np.mean(self.last_episodes_predators_avoided) / self.environment_params["probability_of_predator"] \
                            > self.finished_conditions["PAI"] \
                            and np.mean(self.last_episodes_prey_caught)/self.simulation.available_prey \
                            > self.finished_conditions["PCI"]:
                        print("Final condition surpassed, exiting training...")
                        break

            if self.switch_network_configuration:
                break

            self.save_configuration_files()
            self.episode_loop()
            if self.monitor_performance:
                ps = pstats.Stats(self.profile)
                ps.sort_stats("tottime")
                ps.print_stats(20)
                print(time.time(), flush=True)

                if self.monitor_performance:
                    self.profile = cProfile.Profile()
                    self.profile.enable()

    @staticmethod
    def remove_new_variables(var_list, new_var_names):
        filtered_var_list = []
        for var in var_list:
            if any(new_name in var.name for new_name in new_var_names):
                #print(f"Found in {var.name}")
                pass
            else:
                filtered_var_list.append(var)
        return filtered_var_list

    def create_environment(self):
        if self.continuous_actions:
            self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts,
                                                                self.using_gpu,
                                                                num_actions=self.learning_params["num_actions"])
        else:
            self.simulation = DiscreteNaturalisticEnvironment(self.environment_params, self.realistic_bouts, 
                                                              self.using_gpu, num_actions=self.learning_params["num_actions"])

    def check_update_configuration(self):
        next_point = str(self.configuration_index + 1)
        episode_transition_points = self.episode_transitions.keys()
        switch_criteria_met = False

        print(f"Length of logs: {len(self.last_episodes_prey_caught)}")

        # Switch config by episode
        if next_point in episode_transition_points and self.episode_number > self.episode_transitions[next_point]:
            switch_criteria_met = True
        elif len(self.last_episodes_prey_caught) >= self.min_scaffold_interval:  # Switch config by behavioural conditionals
            prey_conditional_transition_points = self.pci_transitions.keys()
            predators_conditional_transition_points = self.pai_transitions.keys()
            grains_bumped_conditional_transfer_points = self.sgb_transitions.keys()

            if next_point in predators_conditional_transition_points and \
                    np.mean(self.last_episodes_predators_avoided) / self.environment_params["probability_of_predator"] > self.pai_transitions[next_point]:
                switch_criteria_met = True
            elif next_point in prey_conditional_transition_points and \
                    np.mean(self.last_episodes_prey_caught) / self.simulation.available_prey > self.pci_transitions[next_point]:
                switch_criteria_met = True
            elif next_point in grains_bumped_conditional_transfer_points and \
                    np.mean(self.last_episodes_sand_grains_bumped) > self.sgb_transitions[next_point]:
                switch_criteria_met = True

            # if switch_criteria_met:
            #     print("Switch criteria met...")

            # if self.episode_number - self.previous_config_switch < self.min_scaffold_interval:
            #     # print(f"Switch min interval not reached: {self.episode_number} {self.previous_config_switch} {self.min_scaffold_interval}")
            #     switch_criteria_met = False

            # Also check whether no improvement in selected metric is present
            if "scaffold_stasis_requirement" in self.learning_params:
                if switch_criteria_met and self.learning_params["scaffold_stasis_requirement"]:
                    # When based on target parameter:
                    # if next_point in predators_conditional_transition_points:
                    #     important_values = np.array(self.last_episodes_predators_avoided) / self.environment_params["probability_of_predator"]
                    # elif next_point in prey_conditional_transition_points:
                    #     important_values = np.array(self.last_episodes_prey_caught) / self.simulation.available_prey
                    # elif next_point in grains_bumped_conditional_transfer_points:
                    #     important_values = np.array(self.last_episodes_sand_grains_bumped)
                    # else:
                    #     important_values = np.array(self.last_episodes_prey_caught) / self.simulation.available_prey

                    # When based on episode reward:
                    important_values = np.array(self.last_episodes_reward)

                    pre_values = np.mean(important_values[:int(self.min_scaffold_interval/2)])
                    post_values = np.mean(important_values[int(self.min_scaffold_interval/2):])
                    overall_std = np.std(important_values)

                    if post_values - pre_values > overall_std / 10:
                        switch_criteria_met = False
                        print(f"""Still improving Reward: Pre: {pre_values} Post: {post_values} Std: {overall_std}""")
                    else:
                        print(f"""Stopped improving Reward: Pre: {pre_values} Post: {post_values} Std: {overall_std}""")

                    # And PCI criterion
                    important_values = np.array(self.last_episodes_prey_caught) / self.simulation.available_prey

                    pre_values = np.mean(important_values[:int(self.min_scaffold_interval/2)])
                    post_values = np.mean(important_values[int(self.min_scaffold_interval/2):])
                    overall_std = np.std(important_values)

                    if post_values - pre_values > overall_std / 10:
                        switch_criteria_met = False
                        print(f"""Still improving PCI: Pre: {pre_values} Post: {post_values} Std: {overall_std}""")
                    else:
                        print(f"""Stopped improving PCI: Pre: {pre_values} Post: {post_values} Std: {overall_std}""")

        if switch_criteria_met:
            if self.algorithm == "DQN":
                self.experience_buffer.reset()
            self.configuration_index = int(next_point)
            self.switched_configuration = True

            if "maintain_state" in self.learning_params:
                if self.learning_params["maintain_state"]:
                    self.save_rnn_state()
            # Save the model
            self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")
            self.checkpoint_steps = self.total_steps
            print("Saved Model")

            print(f"{self.model_id}: Changing configuration to configuration {self.configuration_index}")
            self.current_configuration_location = f"./Configurations/Training-Configs/{self.config_name}/{str(self.configuration_index)}"

            original_base_network_layers = copy.copy([layer for layer in self.learning_params["base_network_layers"].keys()])
            original_modular_network_layers = copy.copy([layer for layer in self.learning_params["modular_network_layers"].keys()])
            original_layers = original_base_network_layers + original_modular_network_layers

            self.learning_params, self.environment_params = self.load_configuration_files()
            self.previous_config_switch = self.episode_number
            self.create_environment()

            self.last_episodes_prey_caught = []
            self.last_episodes_reward = []

            new_base_network_layers = copy.copy(
                [layer for layer in self.learning_params["base_network_layers"].keys()])
            new_modular_network_layers = copy.copy(
                [layer for layer in self.learning_params["modular_network_layers"].keys()])
            new_layers = new_base_network_layers + new_modular_network_layers

            additional_layers = [layer for layer in new_layers if layer not in original_layers]
            removed_layers = [layer for layer in original_layers if layer not in new_layers]

            if self.environment_params["use_dynamic_network"]:
                if self.algorithm == "PPO":
                    self.original_output_layer = self.actor_network.processing_network_output
                elif self.algorithm == "DQN":
                    self.original_output_layer = self.main_QN.processing_network_output
                else:
                    print("Possible error, algorithm incorrectly specified")

            if len(additional_layers) > 0 or len(removed_layers) > 0:
                print("Network changed, recreating...")
                self.switch_network_configuration = True
                self.additional_layers = additional_layers
                self.removed_layers = removed_layers

            # Reset sigma progression
            if self.continuous_actions and self.environment_params["sigma_scaffolding"]:
                self.sigma_total_steps = 0

            if self.learning_params["epsilon_greedy"]:
                self.epsilon = self.learning_params["startE"]
                self.step_drop = (self.learning_params['startE'] - self.learning_params['endE']) / self.learning_params[
                    'anneling_steps']

            # Make sure visual system parameters are updated.
            self.simulation.fish.left_eye.red_photoreceptor_rf_size = self.environment_params["red_photoreceptor_rf_size"]
            self.simulation.fish.left_eye.uv_photoreceptor_rf_size = self.environment_params["uv_photoreceptor_rf_size"]
            self.simulation.fish.left_eye.env_variables = self.environment_params
            self.simulation.fish.left_eye.get_repeated_computations()

            self.simulation.fish.right_eye.red_photoreceptor_rf_size = self.environment_params["red_photoreceptor_rf_size"]
            self.simulation.fish.right_eye.uv_photoreceptor_rf_size = self.environment_params["uv_photoreceptor_rf_size"]
            self.simulation.fish.right_eye.env_variables = self.environment_params
            self.simulation.fish.right_eye.get_repeated_computations()

            self.simulation.board.light_gain = self.environment_params["light_gain"]
            if "light_gradient" in self.environment_params:
                self.simulation.board.light_gradient = self.environment_params["light_gradient"]

            self.simulation.create_current()
        else:
            self.switched_configuration = False

    def load_transitions(self):
        with open(f"Configurations/Training-Configs/{self.config_name}/transitions.json", 'r') as f:
            transitions = json.load(f)
        self.episode_transitions = transitions["Episode"]
        self.pci_transitions = transitions["PCI"]
        self.pai_transitions = transitions["PAI"]
        self.sgb_transitions = transitions["SGB"]
        if "Finished Condition" in transitions:
            self.finished_conditions = transitions["Finished Condition"]

        configurations = list(self.episode_transitions.keys()) + list(self.pci_transitions.keys()) + \
                         list(self.pai_transitions.keys()) + list(self.sgb_transitions.keys())
        self.total_configurations = len(configurations) + 1

    def save_rnn_state(self):
        if self.environment_params["use_dynamic_network"]:
            data = {}
            num_rnns = len(self.init_rnn_state)
            for rnn in range(num_rnns):
                data_1 = {
                    f"rnn_state_{rnn}_1": self.init_rnn_state[rnn][0].tolist(),
                    f"rnn_state_{rnn}_2": self.init_rnn_state[rnn][1].tolist(),
                    f"rnn_state_{rnn}_ref_1": self.init_rnn_state_ref[rnn][0].tolist(),
                    f"rnn_state_{rnn}_ref_2": self.init_rnn_state_ref[rnn][1].tolist(),
                }
                data = {**data, **data_1}
        else:
            data = {
                f"rnn_state_1": self.init_rnn_state[0].tolist(),
                f"rnn_state_2": self.init_rnn_state[1].tolist(),
                f"rnn_state_ref_1": self.init_rnn_state_ref[0].tolist(),
                f"rnn_state_ref_2": self.init_rnn_state_ref[1].tolist(),
            }

        with open(f"{self.model_location}/rnn_state-{self.episode_number}.json", 'w') as f:
            json.dump(data, f, indent=4)


    def _save_episode(self, episode_start_t, total_episode_reward, prey_caught, predators_avoided, sand_grains_bumped,
                      steps_near_vegetation):
        """Saves episode data common to all models"""
        # print(f"{self.model_id} - episode {str(self.episode_number)}: num steps = {str(self.simulation.num_steps)}",
        #       flush=True)

        # # Log the average training time for episodes (when not saved)
        # if not self.save_frames:
        #     self.training_times.append(time() - episode_start_t)
        #     print(np.mean(self.training_times))

        # Keep recent predators caught for configuration change conditionals
        self.last_episodes_prey_caught.append(prey_caught)
        self.last_episodes_predators_avoided.append(predators_avoided)
        self.last_episodes_sand_grains_bumped.append(sand_grains_bumped)
        self.last_episodes_reward.append(total_episode_reward)

        if len(self.last_episodes_prey_caught) > self.min_scaffold_interval:
            self.last_episodes_prey_caught.pop(0)
        if len(self.last_episodes_reward) > self.min_scaffold_interval:
            self.last_episodes_reward.pop(0)
        if len(self.last_episodes_predators_avoided) > self.min_scaffold_interval:
            self.last_episodes_predators_avoided.pop(0)
        if len(self.last_episodes_sand_grains_bumped) > self.min_scaffold_interval:
            self.last_episodes_sand_grains_bumped.pop(0)

        # SUMMARIES
        # Rewards
        episode_summary = tf.Summary(value=[tf.Summary.Value(tag="episode reward", simple_value=total_episode_reward)])
        self.writer.add_summary(episode_summary, self.episode_number)

        #                  Environmental Logs                   #

        # Cause of death
        death = self.simulation.recent_cause_of_death
        if death is None:
            death_int = 0
        elif death == "Predator":
            death_int = 1
        elif death == "Prey-All-Eaten":
            death_int = 2
        elif death == "Starvation":
            death_int = 3
        elif death == "Salt":
            death_int = 4
        else:
            print("Cause of death label wrong")
            death_int = 99
        cause_of_death_summary = tf.Summary(value=[tf.Summary.Value(tag="Cause of Death", simple_value=death_int)])
        self.writer.add_summary(cause_of_death_summary, self.episode_number)

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
        if self.environment_params["prey_num"] != 0 and self.environment_params["sand_grain_num"] != 0:
            # prey_capture_index = prey_caught / self.environment_params["prey_num"]
            prey_capture_index = prey_caught / self.simulation.available_prey
            sand_grain_capture_index = sand_grains_bumped / self.environment_params["sand_grain_num"]
            # Note, generally would expect to prefer sand grains as each bump counts as a capture.
            if sand_grain_capture_index == 0:
                prey_preference = 1
            else:
                prey_preference = prey_capture_index / (sand_grain_capture_index + prey_capture_index)
                prey_preference -= 0.5
                prey_preference *= 2
            prey_preference_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Prey-Sand-Grain-Preference", simple_value=prey_preference)])
            self.writer.add_summary(prey_preference_summary, self.episode_number)

        if self.environment_params["prey_num"] != 0:
            # fraction_prey_caught = prey_caught / self.environment_params["prey_num"]
            fraction_prey_caught = prey_caught / self.simulation.available_prey
            prey_caught_summary = tf.Summary(
                value=[tf.Summary.Value(tag="prey capture index (fraction caught)", simple_value=fraction_prey_caught)])
            self.writer.add_summary(prey_caught_summary, self.episode_number)

            prey_capture_rate = fraction_prey_caught / self.simulation.num_steps
            prey_capture_rate_summary = tf.Summary(
                value=[tf.Summary.Value(tag="prey capture rate (fraction caught per step)", simple_value=prey_capture_rate)])
            self.writer.add_summary(prey_capture_rate_summary, self.episode_number)

            if (prey_caught + self.simulation.failed_capture_attempts) != 0:

                capture_success_rate = prey_caught / (prey_caught + self.simulation.failed_capture_attempts)
                capture_success_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="capture success rate", simple_value=capture_success_rate)])
                self.writer.add_summary(capture_success_summary, self.episode_number)

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

        if self.environment_params["salt"]:
            mean_salt_damage_per_step = np.mean(self.simulation.salt_damage_history)
            salt_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Mean salt damage taken per step",
                                        simple_value=mean_salt_damage_per_step)])
            self.writer.add_summary(salt_summary, self.episode_number)

        if self.environment_params["dark_light_ratio"] > 0:
            light_dominance = 0.5 / (1-self.environment_params["dark_light_ratio"])
            dark_discount = 0.5 / (self.environment_params["dark_light_ratio"])
            steps_in_light = np.sum((np.array(self.simulation.in_light_history) > 0) * 1)
            steps_in_light_d = steps_in_light * light_dominance
            steps_in_dark = self.simulation.num_steps - steps_in_light
            steps_in_dark_d = steps_in_dark * dark_discount
            fraction_in_light_normalised = steps_in_light_d/(steps_in_dark_d+steps_in_light_d)
            phototaxis_index = (fraction_in_light_normalised-0.5) * 2
            phototaxis_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Phototaxis Index",
                                        simple_value=phototaxis_index)])
            self.writer.add_summary(phototaxis_summary, self.episode_number)

        # Current opposition log
        if self.environment_params["current_setting"] is not False:
            current_opposition_metric = np.sum(self.simulation.vector_agreement)
            # Should be positive when fish swims with current, negative when swims against, and zero if has no preference.
            current_opposition_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Current Opposition",
                                        simple_value=current_opposition_metric)])
            self.writer.add_summary(current_opposition_summary, self.episode_number)

        # Energy efficiency index - Just the average energy used per step.
        if self.environment_params["energy_state"]:
            energy_used = 0
            for i, e in enumerate(self.simulation.energy_level_log):
                if i == 0 or i == len(self.simulation.energy_level_log)-1:
                    pass
                else:
                    if e < self.simulation.energy_level_log[i-1] and e < self.simulation.energy_level_log[i+1]:
                        energy_used += 1-e
            energy_used += 1-self.simulation.energy_level_log[-1]
            energy_efficiency = energy_used/len(self.simulation.energy_level_log)

            energy_efficiency_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Energy Efficiency Index",
                                        simple_value=energy_efficiency)])
            self.writer.add_summary(energy_efficiency_summary, self.episode_number)

        # Exploration index
        fish_positions = np.array(self.simulation.position_buffer) / 100
        fish_positions = np.around(fish_positions).astype(int)
        grid = np.zeros((int(self.environment_params["width"]/100)+1, int(self.environment_params["height"]/100)+1))
        for p in fish_positions:
            grid[p] += 1
        vals = grid[(grid > 0)]
        vals /= fish_positions.shape[0]
        vals = 1 / vals
        exploration_quotient = np.sum(vals)
        exploration_summary = tf.Summary(
            value=[tf.Summary.Value(tag="Exploration Quotient",
                                    simple_value=exploration_quotient)])
        self.writer.add_summary(exploration_summary, self.episode_number)

        if self.switched_configuration:
            configuration_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Configuration change", simple_value=self.configuration_index)]
            )
            self.writer.add_summary(configuration_summary, self.episode_number)

        if self.full_logs:
            episode_duration_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Episode Duration", simple_value=self.simulation.num_steps)]
            )
            self.writer.add_summary(episode_duration_summary, self.episode_number)

        # Periodically save the model.
        # if self.episode_number % self.learning_params['network_saving_frequency'] == 0:
        # checkpoint = tf.train.get_checkpoint_state(self.model_location)
        # if hasattr(checkpoint, "model_checkpoint_path"):
        #     checkpoint_path = checkpoint.model_checkpoint_path
        #     checkpoint_steps = re.sub('\D', '', checkpoint_path)
        # else:
        #     checkpoint_steps = self.learning_params['network_saving_frequency_steps']

        # print(f"CHK: {checkpoint_steps}")
        # print(f"Total steps: {self.total_steps}")

        if "network_saving_frequency_steps" in self.learning_params:
            if self.total_steps - int(self.checkpoint_steps) >= self.learning_params['network_saving_frequency_steps']:

                # IF the steps interval is sufficient, will save the network according to episode number, so matches rnn
                # state and episode number initialisation
                # print(f"mean time: {np.mean(self.training_times)}")
                if self.learning_params["maintain_state"]:
                    self.save_rnn_state()
                # Save the model
                self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")
                self.checkpoint_steps = self.total_steps
                print("Saved Model")
        else:
            if self.episode_number % self.learning_params["summaryLength"] == 0 and self.episode_number != 0:
                # if self.learning_params["maintain_state"]:
                #     self.save_rnn_state()
                # Save the model
                self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")

        if self.episode_number % self.learning_params['summaryLength'] == 0 and self.episode_number != 0:
            if self.using_gpu:
                background = self.simulation.board.global_background_grating.get()[:, :, 0]
            else:
                background = self.simulation.board.global_background_grating[:, :, 0]
            if self.environment_params["salt"]:
                salt_location = self.simulation.salt_location
            else:
                salt_location = None
            internal_state_order = self.get_internal_state_order()

            self.buffer.save_assay_data(f"Episode {self.episode_number}",
                                        self.model_location + "/episodes",
                                        f"Episode {self.episode_number}",
                                        internal_state_order=internal_state_order,
                                        background=background,
                                        salt_location=salt_location)
            episode_data = load_data(f"{self.model_name}-{self.model_number}", f"Episode {self.episode_number}",
                                     f"Episode {self.episode_number}", training_data=True)

            draw_episode(episode_data, self.environment_params, f"{self.model_location}/episodes/Episode {self.episode_number}",
                         self.continuous_actions)

            self.buffer.reset()
            self.save_environmental_data = False

        if (self.episode_number + 1) % self.learning_params['summaryLength'] == 0:
            print('starting to log data', flush=True)
            self.save_environmental_data = True
            #self.buffer.init_assay_recordings(["environmental positions", "observation", "internal state"], [])

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

        # Action Diversity


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

        if self.full_logs:
            # Save Loss

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

    def save_configuration_files(self):
        with open(f"{self.model_location}/learning_configuration.json", 'w') as f:
            json.dump(self.learning_params, f, indent=4)
        with open(f"{self.model_location}/environment_configuration.json", 'w') as f:
            json.dump(self.environment_params, f, indent=4)
