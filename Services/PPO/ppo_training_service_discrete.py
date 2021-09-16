from time import time

import numpy as np
import tensorflow.compat.v1 as tf

from Buffers.ppo_buffer_discrete import PPOBufferDiscrete

from Services.PPO.discrete_ppo import DiscretePPO
from Services.training_service import TrainingService

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_training_target_discrete(trial, total_steps, episode_number, memory_fraction, configuration_index):
    services = PPOTrainingServiceDiscrete(model_name=trial["Model Name"],
                                          trial_number=trial["Trial Number"],
                                          total_steps=total_steps,
                                          episode_number=episode_number,
                                          monitor_gpu=trial["monitor gpu"],
                                          using_gpu=trial["Using GPU"],
                                          memory_fraction=memory_fraction,
                                          config_name=trial["Environment Name"],
                                          realistic_bouts=trial["Realistic Bouts"],
                                          continuous_actions=trial["Continuous Actions"],

                                          model_exists=trial["Model Exists"],
                                          episode_transitions=trial["Episode Transitions"],
                                          total_configurations=trial["Total Configurations"],
                                          conditional_transitions=trial["Conditional Transitions"],
                                          configuration_index=configuration_index,
                                          full_logs=trial["Full Logs"]
                                          )
    services.run()


class PPOTrainingServiceDiscrete(TrainingService, DiscretePPO):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, model_exists, episode_transitions,
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

        self.batch_size = self.learning_params["batch_size"]
        self.trace_length = self.learning_params["trace_length"]
        self.step_drop = (self.learning_params['startE'] - self.learning_params['endE']) / self.learning_params[
            'anneling_steps']

        self.buffer = PPOBufferDiscrete(gamma=0.99, lmbda=0.9, batch_size=self.learning_params["batch_size"],
                                        train_length=self.learning_params["trace_length"], assay=False, debug=False)
        # self.e = self.learning_params["startE"]

    def run(self):
        sess = self.create_session()
        with sess as self.sess:
            self.create_network()
            self.init_states()
            TrainingService._run(self)

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
        self.train_network()

        # Add the episode to tensorflow logs
        self.save_episode(episode_start_t=t0,
                          total_episode_reward=self.total_episode_reward,
                          prey_caught=self.simulation.prey_caught,
                          predators_avoided=self.simulation.predators_avoided,
                          sand_grains_bumped=self.simulation.sand_grains_bumped,
                          steps_near_vegetation=self.simulation.steps_near_vegetation,
                          )

        if self.e > self.learning_params['endE']:
            self.e -= self.step_drop

        print(f"""Total episode reward: {self.total_episode_reward}\n""")

    def step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                  rnn_state_critic_ref):
        if self.full_logs:
            return self._training_step_loop_full_logs(o, internal_state, a, rnn_state_actor,
                                                      rnn_state_actor_ref, rnn_state_critic,
                                                      rnn_state_critic_ref)
        else:
            return self._training_step_loop_reduced_logs(o, internal_state, a, rnn_state_actor,
                                                         rnn_state_actor_ref, rnn_state_critic,
                                                         rnn_state_critic_ref)

    def save_episode(self, episode_start_t, total_episode_reward, prey_caught,
                     predators_avoided, sand_grains_bumped, steps_near_vegetation):
        """
        Saves the episode the the experience buffer. Also creates a gif if at interval.
        """

        TrainingService._save_episode(self, episode_start_t, total_episode_reward, prey_caught,
                                      predators_avoided, sand_grains_bumped, steps_near_vegetation)

        # Action Summary
        for act in range(self.learning_params['num_actions']):
            action_freq = np.sum(np.array(self.buffer.action_buffer) == act) / len(self.buffer.action_buffer)
            a_freq = tf.Summary(value=[tf.Summary.Value(tag="action " + str(act), simple_value=action_freq)])
            self.writer.add_summary(a_freq, self.total_steps)

        # Value Summary
        for step in range(0, len(self.buffer.value_buffer)):
            value_summary = tf.Summary(
                value=[tf.Summary.Value(tag="value_predictions", simple_value=self.buffer.value_buffer[step])])
            self.writer.add_summary(value_summary, self.total_steps - len(self.buffer.value_buffer) + step)

        if self.full_logs:
            # Save Loss
            for step in range(0, len(self.buffer.critic_loss_buffer)):
                critic_loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="critic loss", simple_value=self.buffer.critic_loss_buffer[step])])
                self.writer.add_summary(critic_loss_summary,
                                        self.total_steps - len(self.buffer.action_buffer) + step * self.learning_params[
                                            "batch_size"])
            for step in range(0, len(self.buffer.actor_loss_buffer)):
                actor_loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="critic loss", simple_value=self.buffer.actor_loss_buffer[step])])
                self.writer.add_summary(actor_loss_summary,
                                        self.total_steps - len(self.buffer.action_buffer) + step * self.learning_params[
                                            "batch_size"])