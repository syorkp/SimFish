from time import time
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf

from Buffers.experience_buffer import ExperienceBuffer
from Tools.make_gif import make_gif
from Services.training_service import TrainingService
from Services.DQN.base_dqn import BaseDQN

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def training_target(trial, epsilon, total_steps, episode_number, memory_fraction, configuration_index):
    services = DQNTrainingService(model_name=trial["Model Name"],
                                  trial_number=trial["Trial Number"],
                                  total_steps=total_steps,
                                  episode_number=episode_number,
                                  monitor_gpu=trial["monitor gpu"],
                                  using_gpu=trial["Using GPU"],
                                  memory_fraction=memory_fraction,
                                  config_name=trial["Environment Name"],
                                  realistic_bouts=trial["Realistic Bouts"],
                                  continuous_actions=trial["Continuous Actions"],
                                  epsilon=epsilon,

                                  model_exists=trial["Model Exists"],
                                  episode_transitions=trial["Episode Transitions"],
                                  total_configurations=trial["Total Configurations"],
                                  conditional_transitions=trial["Conditional Transitions"],
                                  configuration_index=configuration_index,
                                  full_logs=trial["Full Logs"]
                                  )
    services.run()


class DQNTrainingService(TrainingService, BaseDQN):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, epsilon, model_exists, episode_transitions,
                 total_configurations, conditional_transitions, configuration_index, full_logs):
        super().__init__(model_name=model_name, trial_number=trial_number,
                         total_steps=total_steps, episode_number=episode_number,
                         monitor_gpu=monitor_gpu, using_gpu=using_gpu,
                         memory_fraction=memory_fraction, config_name=config_name,
                         realistic_bouts=realistic_bouts,
                         continuous_actions=continuous_actions,
                         model_exists=model_exists,
                         episode_transitions=episode_transitions,
                         total_configurations=total_configurations,
                         conditional_transitions=conditional_transitions,
                         configuration_index=configuration_index,
                         full_logs=full_logs)

        self.algorithm = "DQN"
        self.batch_size = self.learning_params["batch_size"]
        self.trace_length = self.learning_params["trace_length"]
        self.step_drop = (self.learning_params['startE'] - self.learning_params['endE']) / self.learning_params[
            'anneling_steps']
        self.pre_train_steps = self.total_steps + self.learning_params["pre_train_steps"]

        # TODO: Check below dont already exist
        self.writer = None
        self.trainables = None
        self.target_ops = None

        # Maintain variables
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = self.learning_params["startE"]

        self.buffer = ExperienceBuffer(output_location=self.model_location, buffer_size=self.learning_params["exp_buffer_size"])

    def _run(self):
        self.create_network()
        self.init_states()
        TrainingService._run(self)


        # Print saved metrics
        # print(f"Total training time: {sum(self.training_times)}")
        # print(f"Total reward: {sum(self.reward_list)}")

    def episode_loop(self):
        t0 = time()
        self.current_episode_max_duration = self.learning_params["max_epLength"]
        all_actions, total_episode_reward, episode_buffer = BaseDQN.episode_loop(self)

        self.save_episode(episode_start_t=t0,
                          all_actions=all_actions,
                          total_episode_reward=total_episode_reward,
                          episode_buffer=episode_buffer,
                          prey_caught=self.simulation.prey_caught,
                          predators_avoided=self.simulation.predators_avoided,
                          sand_grains_bumped=self.simulation.sand_grains_bumped,
                          steps_near_vegetation=self.simulation.steps_near_vegetation
                          )

    def save_episode(self, episode_start_t, all_actions, total_episode_reward, episode_buffer, prey_caught,
                     predators_avoided, sand_grains_bumped, steps_near_vegetation):
        """
        Saves the episode the the experience buffer. Also creates a gif if at interval.
        :param episode_start_t: The time at the start of the episode, used to calculate the time the episode took.
        :param all_actions: The array of all the actions taken during the episode.
        :param total_episode_reward: The total reward of the episode.
        :param episode_buffer: A buffer containing all the state transitions, actions and associated rewards yielded by
        the environment.
        :return:
        """

        TrainingService._save_episode(self, episode_start_t, total_episode_reward, prey_caught,
                                      predators_avoided, sand_grains_bumped, steps_near_vegetation)

        for act in range(self.learning_params['num_actions']):
            action_freq = np.sum(np.array(all_actions) == act) / len(all_actions)
            a_freq = tf.Summary(value=[tf.Summary.Value(tag="action " + str(act), simple_value=action_freq)])
            self.writer.add_summary(a_freq, self.total_steps)

        # Save the parameters to be carried over.
        output_data = {"epsilon": self.epsilon, "episode_number": self.episode_number, "total_steps": self.total_steps, "configuration_index": self.configuration_index}
        with open(f"{self.model_location}/saved_parameters.json", "w") as file:
            json.dump(output_data, file)

        buffer_array = np.array(episode_buffer)
        episode_buffer = list(zip(buffer_array))
        self.buffer.add(episode_buffer)
        # Periodically save the model.
