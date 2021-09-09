from time import time

import numpy as np
import tensorflow.compat.v1 as tf

from Buffers.ppo_buffer import PPOBuffer

from Services.PPO.continuous_ppo import ContinuousPPO
from Services.training_service import TrainingService

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_training_target_continuous(trial, total_steps, episode_number, memory_fraction):
    services = PPOTrainingServiceContinuous(model_name=trial["Model Name"],
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
                                            full_logs=trial["Full Logs"]
                                            )
    services.run()


class PPOTrainingServiceContinuous(TrainingService, ContinuousPPO):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, model_exists, episode_transitions,
                 total_configurations, conditional_transitions, full_logs):
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
                         full_logs=full_logs)

        self.batch_size = self.learning_params["batch_size"]  # TODO: replace all readings with these
        self.trace_length = self.learning_params["trace_length"]

        self.buffer = PPOBuffer(gamma=0.99, lmbda=0.9, batch_size=self.learning_params["batch_size"],
                                train_length=self.learning_params["trace_length"], assay=False, debug=False)

        # TODO: Move this to TrainingService

    def _run(self):
        self.create_network()
        TrainingService._run(self)

    def episode_loop(self):
        """
        Loops over an episode, which involves initialisation of the environment and RNN state, then iteration over the
        steps in the episode. The relevant values are then saved to the experience buffer.
        """
        t0 = time()

        self.current_episode_max_duration = self.learning_params["max_epLength"]
        ContinuousPPO.episode_loop(self)

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

        print(f"""Mean Impulse: {np.mean([i[0] for i in self.buffer.action_buffer])}
Mean Angle {np.mean([i[1] for i in self.buffer.action_buffer])}
Total episode reward: {self.total_episode_reward}\n""")

    def step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                  rnn_state_critic_ref):
        if self.full_logs:
            return ContinuousPPO._training_step_loop_full_logs(self, o, internal_state, a, rnn_state_actor,
                                                               rnn_state_actor_ref, rnn_state_critic,
                                                               rnn_state_critic_ref)
        else:
            return ContinuousPPO._training_step_loop_reduced_logs(self, o, internal_state, a, rnn_state_actor,
                                                                  rnn_state_actor_ref, rnn_state_critic,
                                                                  rnn_state_critic_ref)

    def save_episode(self, episode_start_t, total_episode_reward, prey_caught,
                     predators_avoided, sand_grains_bumped, steps_near_vegetation):
        """
        Saves the episode the the experience buffer. Also creates a gif if at interval.
        """
        # TODO: Make these neater

        TrainingService._save_episode(self, episode_start_t, total_episode_reward, prey_caught,
                                      predators_avoided, sand_grains_bumped, steps_near_vegetation)

        # Action Summary
        impulses = [action[0] for action in self.buffer.action_buffer]
        for step in range(0, len(impulses), 5):
            impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="impulse magnitude", simple_value=impulses[step])])
            self.writer.add_summary(impulse_summary, self.total_steps - len(impulses) + step)

        angles = [action[1] for action in self.buffer.action_buffer]
        for step in range(0, len(angles), 5):
            angles_summary = tf.Summary(value=[tf.Summary.Value(tag="angle magnitude", simple_value=angles[step])])
            self.writer.add_summary(angles_summary, self.total_steps - len(angles) + step)

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
                                        self.total_steps - len(angles) + step * self.learning_params["batch_size"])

            for step in range(0, len(self.buffer.impulse_loss_buffer)):
                impulse_loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="impulse loss", simple_value=self.buffer.impulse_loss_buffer[step])])
                self.writer.add_summary(impulse_loss_summary,
                                        self.total_steps - len(angles) + step * self.learning_params["batch_size"])

            for step in range(0, len(self.buffer.angle_loss_buffer)):
                angle_loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="angle loss", simple_value=self.buffer.angle_loss_buffer[step])])
                self.writer.add_summary(angle_loss_summary,
                                        self.total_steps - len(angles) + step * self.learning_params["batch_size"])

            # Saving Parameters for Testing
            for step in range(0, len(self.buffer.mu_i_buffer)):
                mu_i_loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="mu_impulse", simple_value=self.buffer.mu_i_buffer[step])])
                self.writer.add_summary(mu_i_loss_summary, self.total_steps - len(self.buffer.mu_i_buffer) + step)

            for step in range(0, len(self.buffer.si_i_buffer)):
                si_i_loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="sigma_impulse", simple_value=self.buffer.si_i_buffer[step])])
                self.writer.add_summary(si_i_loss_summary, self.total_steps - len(self.buffer.si_i_buffer) + step)

            for step in range(0, len(self.buffer.mu_a_buffer)):
                mu_a_loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="mu_angle", simple_value=self.buffer.mu_a_buffer[step])])
                self.writer.add_summary(mu_a_loss_summary, self.total_steps - len(self.buffer.mu_a_buffer) + step)

            for step in range(0, len(self.buffer.si_a_buffer)):
                si_a_loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="sigma_angle", simple_value=self.buffer.si_a_buffer[step])])
                self.writer.add_summary(si_a_loss_summary, self.total_steps - len(self.buffer.si_a_buffer) + step)

            for step in range(0, len(self.buffer.mu1_buffer)):
                mu1_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="mu_impulse_base", simple_value=self.buffer.mu1_buffer[step])])
                self.writer.add_summary(mu1_summary, self.total_steps - len(self.buffer.mu1_buffer) + step)

            for step in range(0, len(self.buffer.mu1_ref_buffer)):
                mu1_ref_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="mu_impulse_ref_base", simple_value=self.buffer.mu1_ref_buffer[step])])
                self.writer.add_summary(mu1_ref_summary, self.total_steps - len(self.buffer.mu1_ref_buffer) + step)

            for step in range(0, len(self.buffer.mu_a1_buffer)):
                mu1_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="mu_angle_base", simple_value=self.buffer.mu_a1_buffer[step])])
                self.writer.add_summary(mu1_summary, self.total_steps - len(self.buffer.mu_a1_buffer) + step)

            for step in range(0, len(self.buffer.mu_a_ref_buffer)):
                mu1_ref_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="mu_angle_ref_base", simple_value=self.buffer.mu_a_ref_buffer[step])])
                self.writer.add_summary(mu1_ref_summary, self.total_steps - len(self.buffer.mu_a_ref_buffer) + step)
