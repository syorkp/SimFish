from time import time
import json

import numpy as np

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Analysis.Indexing.data_index_service import DataIndexServiceDiscrete
from Analysis.Behavioural.Exploration.turn_chain_metric import get_normalised_turn_chain_metric_discrete
from Buffers.DQN.dqn_training_buffer import DQNTrainingBuffer
from Buffers.DQN.dqn_assay_buffer import DQNAssayBuffer
from Configurations.Templates.assay_config import naturalistic_assay_config
from Configurations.Utilities.turn_model_configs_into_assay_configs import transfer_config
from Services.training_service import TrainingService
from Services.DQN.dqn_assay_service import assay_target
from Services.DQN.base_dqn import BaseDQN

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def training_target(trial, epsilon, total_steps, episode_number, memory_fraction, configuration_index):
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
        continuous_actions = False

    if "Full Logs" in trial:
        full_logs = trial["Full Logs"]
    else:
        full_logs = True

    if "Profile Speed" in trial:
        profile_speed = trial["Profile Speed"]
    else:
        profile_speed = False


    services = DQNTrainingService(model_name=trial["Model Name"],
                                  trial_number=trial["Trial Number"],
                                  total_steps=total_steps,
                                  episode_number=episode_number,
                                  monitor_gpu=monitor_gpu,
                                  using_gpu=using_gpu,
                                  memory_fraction=memory_fraction,
                                  config_name=trial["Environment Name"],
                                  continuous_actions=continuous_actions,
                                  epsilon=epsilon,
                                  model_exists=trial["Model Exists"],
                                  configuration_index=configuration_index,
                                  full_logs=full_logs,
                                  profile_speed=profile_speed,
                                  )
    services.run()


class DQNTrainingService(TrainingService, BaseDQN):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, continuous_actions, epsilon, model_exists, configuration_index, full_logs, profile_speed):
        super().__init__(model_name=model_name, trial_number=trial_number,
                         total_steps=total_steps, episode_number=episode_number,
                         monitor_gpu=monitor_gpu, using_gpu=using_gpu,
                         memory_fraction=memory_fraction, config_name=config_name,
                         continuous_actions=continuous_actions,
                         model_exists=model_exists,
                         configuration_index=configuration_index,
                         full_logs=full_logs,
                         profile_speed=profile_speed
                         )

        self.algorithm = "DQN"
        self.batch_size = self.learning_params["batch_size"]
        self.trace_length = self.learning_params["trace_length"]
        self.step_drop = (self.learning_params['startE'] - self.learning_params['endE']) / self.learning_params[
            'anneling_steps']
        self.pre_train_steps = self.total_steps + self.learning_params["pre_train_steps"]  # To prevent training following loading if no buffer.
        self.initial_exploration_steps = self.learning_params["pre_train_steps"]  # To allow an initial period of exploration, not repeated upon loading.
        self.last_position_dim = self.environment_params["prey_num"]

        self.writer = None
        self.trainables = None
        self.target_ops = None
        self.action_usage = np.zeros(self.learning_params["num_actions"])

        # Maintain variables
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = self.learning_params["startE"]

        self.experience_buffer = DQNTrainingBuffer(output_location=self.model_location,
                                                   buffer_size=self.learning_params["exp_buffer_size"])

        self.buffer = DQNAssayBuffer()

    def run(self):
        sess = self.create_session()
        with sess as self.sess:
            self.create_network()
            self.init_states()
            TrainingService._run(self)
            self.saver.save(self.sess, f"{self.model_location}/model-{str(self.episode_number)}.cptk")
            # Save the parameters to be carried over.
            output_data = {"epsilon": self.epsilon,
                           "episode_number": self.episode_number,
                           "total_steps": self.total_steps,
                           "configuration_index": self.configuration_index
                           }
            with open(f"{self.model_location}/saved_parameters.json", "w") as file:
                json.dump(output_data, file)

        while self.switch_network_configuration:
            tf.reset_default_graph()
            sess = self.create_session()
            print("Switching network configuration...")
            with sess as self.sess:
                self.create_network()
                new_output_layer = self.main_QN.processing_network_output

                if new_output_layer != self.original_output_layer:  # If altered shape of final output layer
                    self.additional_layers += ["targetaw", "mainaw", "mainvw", "targetvw"]

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
        trial["Continuous Actions"] = False
        trial["Learning Algorithm"] = "DQN"
        for i, assay in enumerate(trial["Assays"]):
            trial["Assays"][i]["duration"] = self.learning_params["max_epLength"]
            trial["Assays"][i]["save frames"] = False

        # Run data gathering
        assay_target(trial, self.total_steps, self.episode_number, self.memory_fraction)

        # Perform cursory analysis on data
        data_index_service = DataIndexServiceDiscrete(self.model_id)
        data_index_service.produce_behavioural_summary_display()

    def episode_loop(self):
        t0 = time()
        self.current_episode_max_duration = self.learning_params["max_epLength"]
        all_actions, total_episode_reward, experience = BaseDQN.episode_loop(self)

        self.save_episode(episode_start_t=t0,
                          all_actions=all_actions,
                          total_episode_reward=total_episode_reward,
                          experience=experience,
                          prey_caught=self.simulation.prey_caught,
                          predators_avoided=self.simulation.predator_attacks_avoided,
                          sand_grains_bumped=self.simulation.sand_grains_bumped,
                          )
        print(f"""{self.model_id} - episode {str(self.episode_number)}: num steps = {str(self.simulation.num_steps)}
Total episode reward: {total_episode_reward}\n""", flush=True)

    def save_episode(self, episode_start_t, all_actions, total_episode_reward, experience, prey_caught,
                     predators_avoided, sand_grains_bumped):
        """
        Saves the episode the experience buffer.
        :param prey_caught:
        :param sand_grains_bumped:
        :param predators_avoided:
        :param episode_start_t: The time at the start of the episode, used to calculate the time the episode took.
        :param all_actions: The array of all the actions taken during the episode.
        :param total_episode_reward: The total reward of the episode.
        :param episode_buffer: A buffer containing all the state transitions, actions and associated rewards yielded by
        the environment.
        :return:
        """

        TrainingService._save_episode(self, episode_start_t, total_episode_reward, prey_caught,
                                      predators_avoided, sand_grains_bumped)

        # Action Diversity
        all_actions_frequency = []

        for act in range(self.learning_params['num_actions']):
            action_freq = np.sum(np.array(all_actions) == act) / len(all_actions)
            a_freq = tf.Summary(value=[tf.Summary.Value(tag="action " + str(act), simple_value=action_freq)])
            self.writer.add_summary(a_freq, self.episode_number)
            all_actions_frequency.append(np.sum(np.array(all_actions) == act) )
        all_actions_frequency = np.array(all_actions_frequency)

        # Normalise given current epsilon value (subtract expected random actions from each group, then clip to zero)
        if self.total_steps > self.pre_train_steps:
            expected_random_actions = (self.epsilon * self.simulation.num_steps)/self.learning_params['num_actions']
            all_actions_frequency = all_actions_frequency.astype(float)
            all_actions_frequency -= expected_random_actions
            all_actions_frequency = np.clip(all_actions_frequency, 0, self.total_steps)
            max_freq_diffs = [np.max(np.absolute([f - f2 for f2 in all_actions_frequency])) for f in all_actions_frequency]
            heterogeneity_score = self.learning_params["num_actions"]/np.sum(max_freq_diffs) - 1/np.sum(all_actions_frequency)
            a_freq = tf.Summary(value=[tf.Summary.Value(tag="Action Heterogeneity Score", simple_value=heterogeneity_score)])
            self.writer.add_summary(a_freq, self.episode_number)

        # Turn chain metric
        turn_chain_summary = tf.Summary(value=[tf.Summary.Value(tag="turn chain preference",
                                                                simple_value=get_normalised_turn_chain_metric_discrete(all_actions))])
        self.writer.add_summary(turn_chain_summary, self.episode_number)

        buffer_array = np.array(experience)
        experience = list(zip(buffer_array))
        self.experience_buffer.add(experience)
