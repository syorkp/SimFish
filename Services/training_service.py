import json
import time
import numpy as np
import pstats
import cProfile
import gc

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment

from Networks.DQN.graph_functions import update_target_graph, update_target

from Services.base_service import BaseService


tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class TrainingService(BaseService):

    def __init__(self, model_name, trial_number, total_steps, episode_number, using_gpu, memory_fraction,
                 config_name, continuous_actions, model_exists, configuration_index, profile_speed):

        super().__init__(model_name, trial_number, total_steps, episode_number, using_gpu, memory_fraction,
                         config_name, continuous_actions, profile_speed)

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

        # For config scaffolding
        self.previous_config_switch = self.episode_number

        self.last_episodes_prey_caught = []
        self.last_episodes_reward = []
        self.last_episodes_predators_avoided = []
        self.last_episodes_sand_grains_bumped = []

        # Determining whether RNN exists in network
        # rnns = [layer for layer in self.learning_params["base_network_layers"].keys() if
        #         self.learning_params["base_network_layers"][layer][0] == "dynamic_rnn"] + \
        #        [layer for layer in self.learning_params["modular_network_layers"].keys() if
        #         self.learning_params["modular_network_layers"][layer][0] == "dynamic_rnn"]
        self.rnn_in_network = True # if len(rnns) > 0 else False

        # For regular saving
        self.save_environmental_data = False

        if "min_scaffold_interval" in self.learning_params:
            self.min_scaffold_interval = self.learning_params["min_scaffold_interval"]
        else:
            self.min_scaffold_interval = 20

        if not hasattr(self, "experience_buffer"):
            self.experience_buffer = None

    def _run(self):
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
                    if np.mean(self.last_episodes_predators_avoided) / self.environment_params["probability_of_predator"] \
                            > self.finished_conditions["PAI"] \
                            and np.mean(self.last_episodes_prey_caught)/self.simulation.available_prey \
                            > self.finished_conditions["PCI"]:
                        print("Final condition surpassed, exiting training...")
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

    def create_environment(self):
        if self.continuous_actions:
            self.simulation = ContinuousNaturalisticEnvironment(self.environment_params,
                                                                self.using_gpu,
                                                                num_actions=self.learning_params["num_actions"],
                                                                )
        else:
            self.simulation = DiscreteNaturalisticEnvironment(self.environment_params,
                                                              self.using_gpu,
                                                              num_actions=self.learning_params["num_actions"],
                                                              )

    def check_update_configuration(self):
        """Check whether the specified performance criteria to switch configurations have been met."""

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

            # Also check whether no improvement in selected metric is present
            if "scaffold_stasis_requirement" in self.learning_params:
                if switch_criteria_met and self.learning_params["scaffold_stasis_requirement"]:
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

            self.save_rnn_state()

            # Save the model
            self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")
            self.checkpoint_steps = self.total_steps
            print("Saved Model")

            print(f"{self.model_id}: Changing configuration to configuration {self.configuration_index}")
            self.current_configuration_location = f"./Configurations/Training-Configs/{self.config_name}/{str(self.configuration_index)}"

            self.learning_params, self.environment_params = self.load_configuration_files()
            self.previous_config_switch = self.episode_number
            self.create_environment()

            self.last_episodes_prey_caught = []
            self.last_episodes_reward = []

            # Reset sigma progression
            if self.continuous_actions:
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

            gc.collect()
        else:
            self.switched_configuration = False

    def load_transitions(self):
        """Load the transitions.json file that specify the conditions under which the scaffold should switch."""

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
        """Save the current RNN state to a json file."""

        data = {}
        # num_rnns = len(self.init_rnn_state)
        num_rnns = 1

        if len(self.init_rnn_state) > num_rnns:
            self.init_rnn_state = [self.init_rnn_state]
            self.init_rnn_state_ref = [self.init_rnn_state_ref]

        for rnn in range(num_rnns):
            data_1 = {
                f"rnn_state_{rnn}_1": self.init_rnn_state[rnn][0].tolist(),
                f"rnn_state_{rnn}_2": self.init_rnn_state[rnn][1].tolist(),
                f"rnn_state_{rnn}_ref_1": self.init_rnn_state_ref[rnn][0].tolist(),
                f"rnn_state_{rnn}_ref_2": self.init_rnn_state_ref[rnn][1].tolist(),
            }
            data = {**data, **data_1}

        with open(f"{self.model_location}/rnn_state-{self.episode_number}.json", 'w') as f:
            json.dump(data, f, indent=4)

    def _save_episode(self, total_episode_reward, prey_caught, sand_grains_bumped):
        """Saves episode data common to all models"""

        self.last_episodes_prey_caught.append(prey_caught)
        self.last_episodes_predators_avoided.append(self.simulation.total_predators_survived)
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
                value=[tf.Summary.Value(tag="predators avoided", simple_value=self.simulation.total_predators_survived)])
            self.writer.add_summary(predators_avoided_summary, self.episode_number)

        if self.environment_params["sand_grain_num"] > 0:
            sand_grains_bumped_summary = tf.Summary(
                value=[tf.Summary.Value(tag="attempted sand grain captures", simple_value=sand_grains_bumped)])
            self.writer.add_summary(sand_grains_bumped_summary, self.episode_number)

        # Normalised Logs
        if self.environment_params["prey_num"] != 0 and self.environment_params["sand_grain_num"] != 0:
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
            if self.simulation.total_predators > 0:
                predator_avoided_index = self.simulation.total_predators_survived / self.simulation.total_predators
            else:
                predator_avoided_index = 0.0
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

        if self.environment_params["salt"]:
            mean_salt_damage_per_step = np.mean(self.simulation.salt_damage_history)
            salt_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Mean salt damage taken per step",
                                        simple_value=mean_salt_damage_per_step)])
            self.writer.add_summary(salt_summary, self.episode_number)

            mean_salt_penalty = self.simulation.salt_associated_reward / self.simulation.num_steps
            salt_penalty_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Mean salt penalty",
                                        simple_value=mean_salt_penalty)])
            self.writer.add_summary(salt_penalty_summary, self.episode_number)

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

            # Should be positive when fish swims with current, negative when swims against, and zero if no preference.
            current_opposition_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Current Opposition",
                                        simple_value=current_opposition_metric)])
            self.writer.add_summary(current_opposition_summary, self.episode_number)

        # Prey availability metric
        prop_steps_prey_available = self.simulation.num_steps_prey_available / self.simulation.num_steps
        prop_steps_prey_available_summary = tf.Summary(
            value=[tf.Summary.Value(tag="Prop Steps Prey Available",
                                    simple_value=prop_steps_prey_available)])
        self.writer.add_summary(prop_steps_prey_available_summary, self.episode_number)

        num_steps_prey_available_summary = tf.Summary(
            value=[tf.Summary.Value(tag="Num Steps Prey Available",
                                    simple_value=self.simulation.num_steps_prey_available)])
        self.writer.add_summary(num_steps_prey_available_summary, self.episode_number)

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
        grid = np.zeros((int(self.environment_params["arena_width"]/100)+1, int(self.environment_params["arena_height"]/100)+1))
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

        episode_duration_summary = tf.Summary(
            value=[tf.Summary.Value(tag="Episode Duration", simple_value=self.simulation.num_steps)]
        )
        self.writer.add_summary(episode_duration_summary, self.episode_number)

        if "network_saving_frequency_steps" in self.learning_params:
            if self.total_steps - int(self.checkpoint_steps) >= self.learning_params['network_saving_frequency_steps']:

                # IF the steps interval is sufficient, will save the network according to episode number, so matches rnn
                # state and episode number initialisation
                self.save_rnn_state()
                # Save the model
                self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")
                self.checkpoint_steps = self.total_steps
                output_data = {"epsilon": self.epsilon, "episode_number": self.episode_number,
                               "total_steps": self.total_steps, "configuration_index": self.configuration_index}
                with open(f"{self.model_location}/saved_parameters.json", "w") as file:
                    json.dump(output_data, file)
                print("Saved Model")
        else:
            if self.episode_number % self.learning_params["summaryLength"] == 0 and self.episode_number != 0:

                # Save the model
                self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")

        if self.episode_number % self.learning_params['summaryLength'] == 0 and self.episode_number != 0:
            if self.using_gpu:
                sediment = self.simulation.board.global_sediment_grating.get()[:, :, 0]
            else:
                sediment = self.simulation.board.global_sediment_grating[:, :, 0]
            if self.environment_params["salt"]:
                salt_location = self.simulation.salt_location
            else:
                salt_location = None
            internal_state_order = self.get_internal_state_order()

            self.buffer.save_assay_data(f"Episode {self.episode_number}",
                                        self.model_location + "/episodes",
                                        f"Episode {self.episode_number}",
                                        internal_state_order=internal_state_order,
                                        sediment=sediment,
                                        salt_location=salt_location)

            self.buffer.reset()
            self.save_environmental_data = False

        if (self.episode_number + 1) % self.learning_params['summaryLength'] == 0:
            print('starting to log data', flush=True)
            self.save_environmental_data = True

        self.reward_list.append(total_episode_reward)

    def save_configuration_files(self):
        """Saves the configuration files for the current scaffold point to the model directory."""
        with open(f"{self.model_location}/learning_configuration.json", 'w') as f:
            json.dump(self.learning_params, f, indent=4)
        with open(f"{self.model_location}/environment_configuration.json", 'w') as f:
            json.dump(self.environment_params, f, indent=4)
