import math
import numpy as np
import copy

import tensorflow.compat.v1 as tf

from Networks.PPO.proximal_policy_optimizer_continuous import PPONetworkActor
from Networks.PPO.proximal_policy_optimizer_continuous_multivariate import PPONetworkActorMultivariate
from Services.PPO.base_ppo import BasePPO

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class ContinuousPPO(BasePPO):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("PPO Constructor called")

        self.continuous = True

        self.impulse_sigma = None
        self.angle_sigma = None
        self.multivariate = None

        self.output_dimensions = 2

    def create_network(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        actor_cell, internal_states = BasePPO.create_network(self)

        if self.multivariate:
            self.actor_network = PPONetworkActorMultivariate(simulation=self.simulation,
                                                             rnn_dim=self.learning_params['rnn_dim_shared'],
                                                             rnn_cell=actor_cell,
                                                             my_scope='actor',
                                                             internal_states=internal_states,
                                                             max_impulse=self.environment_params['max_impulse'],
                                                             max_angle_change=self.environment_params[
                                                                 'max_angle_change'],
                                                             clip_param=self.environment_params['clip_param']
                                                             )

        else:
            self.actor_network = PPONetworkActor(simulation=self.simulation,
                                                 rnn_dim=self.learning_params['rnn_dim_shared'],
                                                 rnn_cell=actor_cell,
                                                 my_scope='actor',
                                                 internal_states=internal_states,
                                                 max_impulse=self.environment_params['max_impulse'],
                                                 max_angle_change=self.environment_params['max_angle_change'],
                                                 clip_param=self.environment_params['clip_param']
                                                 )

    def update_sigmas(self):
        # Exponential scale
        # self.impulse_sigma = np.array([self.env["min_sigma_impulse"] + (
        #             self.env["max_sigma_impulse"] - self.env["min_sigma_impulse"]) * np.e ** (
        #                                            -self.total_steps * self.env["sigma_time_constant"])])
        # self.angle_sigma = np.array([self.env["min_sigma_angle"] + (
        #             self.env["max_sigma_angle"] - self.env["min_sigma_angle"]) * np.e ** (
        #                                          -self.total_steps * self.env["sigma_time_constant"])])

        # Linear scale
        self.impulse_sigma = np.array([self.environment_params["max_sigma_impulse"] - (
                self.environment_params["max_sigma_impulse"] - self.environment_params["min_sigma_impulse"]) * (
                                               self.total_steps / 5000000)])
        self.angle_sigma = np.array([self.environment_params["max_sigma_angle"] - (
                self.environment_params["max_sigma_angle"] - self.environment_params["min_sigma_angle"]) * (
                                             self.total_steps / 5000000)])

    def _episode_loop(self, a=None):
        self.update_sigmas()
        a = [4.0, 0.0]
        super(ContinuousPPO, self)._episode_loop(a)

    def _assay_step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                         rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, conv1l_actor, conv2l_actor, conv3l_actor, \
        conv4l_actor, conv1r_actor, conv2r_actor, conv3r_actor, conv4r_actor, impulse_probability, \
        angle_probability, mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output,
             self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.conv1l, self.actor_network.conv2l, self.actor_network.conv3l,
             self.actor_network.conv4l,
             self.actor_network.conv1r, self.actor_network.conv2r, self.actor_network.conv3r,
             self.actor_network.conv4r,

             self.actor_network.log_prob_impulse, self.actor_network.log_prob_angle,
             self.actor_network.mu_impulse_combined, self.actor_network.sigma_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.sigma_angle_combined, self.actor_network.mu_impulse,
             self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.trainLength: 1,
                       self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined: self.angle_sigma,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref, conv1l_critic, conv2l_critic, conv3l_critic, \
        conv4l_critic, conv1r_critic, conv2r_critic, conv3r_critic, conv4r_critic = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref,
             self.critic_network.conv1l, self.critic_network.conv2l, self.critic_network.conv3l,
             self.critic_network.conv4l,
             self.critic_network.conv1r, self.critic_network.conv2r, self.critic_network.conv3r,
             self.critic_network.conv4r,
             ],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.trainLength: 1,
                       }
        )
        action = [impulse[0][0], angle[0][0]]

        o1, given_reward, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=action,
                                                                                                     frame_buffer=self.frame_buffer,
                                                                                                     save_frames=True,
                                                                                                     activations=(sa,))

        sand_grain_positions, prey_positions, predator_position, vegetation_positions = self.get_positions()  # TODO: Modify

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=given_reward,
                                 value=V,
                                 l_p_impulse=impulse_probability,
                                 l_p_angle=angle_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)

        if "environmental positions" in self.buffer.recordings:
            self.buffer.save_environmental_positions(self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     vegetation_positions,
                                                     self.simulation.fish.body.angle,
                                                     )
        if "convolutional layers" in self.buffer.recordings:
            self.buffer.save_conv_states(conv1l_actor, conv2l_actor, conv3l_actor, conv4l_actor, conv1r_actor,
                                         conv2r_actor, conv3r_actor, conv4r_actor,
                                         conv1l_critic, conv2l_critic, conv3l_critic, conv4l_critic, conv1r_critic,
                                         conv2r_critic, conv3r_critic, conv4r_critic)

        return given_reward, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, \
               updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _training_step_loop(self, actor_network_to_get, actor_network_feed_dict):
        ...

    def _training_step_multivariate_full_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                              rnn_state_critic,
                                              rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, action_probability, mu_i, si_i, mu_a, \
        si_a, mu1, mu1_ref, mu_a1, mu_a_ref = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.log_prob,
             self.actor_network.mu_impulse_combined, self.actor_network.sigma_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.sigma_angle_combined, self.actor_network.mu_impulse, self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.trainLength: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.trainLength: 1,
                       }
        )

        action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=r,
                                 value=V,
                                 l_p_action=action_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _training_step_multivariate_reduced_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                              rnn_state_critic,
                                              rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, action_probability = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.log_prob,
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.trainLength: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.trainLength: 1,
                       }
        )

        action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=r,
                                 value=V,
                                 l_p_action=action_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _training_step_loop_reduced_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                         rnn_state_critic,
                                         rnn_state_critic_ref):
        # Generate actions and corresponding steps.
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, impulse_probability, angle_probability = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.log_prob_impulse, self.actor_network.log_prob_angle,
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined: self.angle_sigma,
                       self.actor_network.batch_size: 1,
                       self.actor_network.trainLength: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.trainLength: 1,
                       }
        )

        action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=r,
                                 value=V,
                                 l_p_impulse=impulse_probability,
                                 l_p_angle=angle_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _training_step_loop_full_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                      rnn_state_critic,
                                      rnn_state_critic_ref):
        # Generate actions and corresponding steps.
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network
        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, impulse_probability, angle_probability, \
        mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.log_prob_impulse, self.actor_network.log_prob_angle,
             self.actor_network.mu_impulse_combined, self.actor_network.sigma_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.sigma_angle_combined, self.actor_network.mu_impulse, self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined: self.angle_sigma,
                       self.actor_network.batch_size: 1,
                       self.actor_network.trainLength: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.trainLength: 1,
                       }
        )

        action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=r,
                                 value=V,
                                 l_p_impulse=impulse_probability,
                                 l_p_angle=angle_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def get_batch_multivariate(self, batch, observation_buffer, internal_state_buffer, action_buffer,
                               previous_action_buffer,
                               log_action_probability_buffer, advantage_buffer, return_buffer):

        observation_batch = observation_buffer[
                            batch * self.batch_size: (batch + 1) * self.batch_size]
        internal_state_batch = internal_state_buffer[
                               batch * self.batch_size: (batch + 1) * self.batch_size]
        action_batch = action_buffer[
                       batch * self.batch_size: (batch + 1) * self.batch_size]
        previous_action_batch = previous_action_buffer[
                                batch * self.batch_size: (batch + 1) * self.batch_size]
        log_action_probability_batch = log_action_probability_buffer[
                                       batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params[
                                           "batch_size"]]
        advantage_batch = advantage_buffer[
                          batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]
        return_batch = return_buffer[
                       batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]

        current_batch_size = observation_batch.shape[0]

        # Stacking for correct network dimensions
        observation_batch = np.vstack(np.vstack(observation_batch))
        internal_state_batch = np.vstack(np.vstack(internal_state_batch))
        action_batch = np.reshape(action_batch[:, :, :],
                                  (self.learning_params["trace_length"] * current_batch_size, 2))
        previous_action_batch = np.vstack(np.vstack(previous_action_batch))
        log_action_probability_batch = log_action_probability_batch.flatten()
        advantage_batch = np.vstack(advantage_batch).flatten()
        return_batch = np.vstack(np.vstack(return_batch)).flatten()

        return observation_batch, internal_state_batch, action_batch, previous_action_batch, \
               log_action_probability_batch, advantage_batch, return_batch, \
               current_batch_size

    def get_batch(self, batch, observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer,
                  log_impulse_probability_buffer, log_angle_probability_buffer, advantage_buffer, return_buffer):

        observation_batch = observation_buffer[
                            batch * self.batch_size: (batch + 1) * self.batch_size]
        internal_state_batch = internal_state_buffer[
                               batch * self.batch_size: (batch + 1) * self.batch_size]
        action_batch = action_buffer[
                       batch * self.batch_size: (batch + 1) * self.batch_size]
        previous_action_batch = previous_action_buffer[
                                batch * self.batch_size: (batch + 1) * self.batch_size]
        log_impulse_probability_batch = log_impulse_probability_buffer[
                                        batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params[
                                            "batch_size"]]
        log_angle_probability_batch = log_angle_probability_buffer[
                                      batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params[
                                          "batch_size"]]
        advantage_batch = advantage_buffer[
                          batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]
        return_batch = return_buffer[
                       batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]

        current_batch_size = observation_batch.shape[0]

        # Stacking for correct network dimensions
        observation_batch = np.vstack(np.vstack(observation_batch))
        internal_state_batch = np.vstack(np.vstack(internal_state_batch))
        impulse_batch = np.reshape(action_batch[:, :, 0],
                                   (self.learning_params["trace_length"] * current_batch_size, 1))
        angle_batch = np.reshape(action_batch[:, :, 1], (self.learning_params["trace_length"] * current_batch_size, 1))
        previous_action_batch = np.vstack(np.vstack(previous_action_batch))
        log_impulse_probability_batch = log_impulse_probability_batch.flatten()
        log_angle_probability_batch = log_angle_probability_batch.flatten()
        advantage_batch = np.vstack(advantage_batch).flatten()
        return_batch = np.vstack(np.vstack(return_batch)).flatten()

        return observation_batch, internal_state_batch, impulse_batch, angle_batch, previous_action_batch, \
               log_impulse_probability_batch, log_angle_probability_batch, advantage_batch, return_batch, \
               current_batch_size

    def train_network(self):
        observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, \
        log_impulse_probability_buffer, log_angle_probability_buffer, advantage_buffer, return_buffer, \
        key_rnn_points = self.buffer.get_episode_buffer()

        number_of_batches = int(math.ceil(observation_buffer.shape[0] / self.learning_params["batch_size"]))

        for batch in range(number_of_batches):
            # Find steps at start of each trace to compute RNN states
            batch_key_points = [i for i in key_rnn_points if
                                batch * self.learning_params["batch_size"] * self.learning_params[
                                    "trace_length"] <= i < (batch + 1) *
                                self.learning_params["batch_size"] * self.learning_params["trace_length"]]

            # Get the current batch
            observation_batch, internal_state_batch, impulse_batch, angle_batch, previous_action_batch, \
            log_impulse_probability_batch, log_angle_probability_batch, advantage_batch, \
            return_batch, current_batch_size = self.get_batch(batch, observation_buffer, internal_state_buffer,
                                                              action_buffer, previous_action_buffer,
                                                              log_impulse_probability_buffer,
                                                              log_angle_probability_buffer, advantage_buffer,
                                                              return_buffer)

            # Loss value logging
            average_loss_value = 0
            average_loss_impulse = 0
            average_loss_angle = 0

            for i in range(self.learning_params["n_updates_per_iteration"]):
                # Compute RNN states for start of each trace.
                actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, \
                critic_rnn_state_ref_slice = self.compute_rnn_states(batch_key_points,
                                                                     observation_buffer[
                                                                     :(batch + 1) * self.learning_params["batch_size"]],
                                                                     internal_state_buffer[
                                                                     :(batch + 1) * self.learning_params["batch_size"]],
                                                                     previous_action_buffer[
                                                                     :(batch + 1) * self.learning_params[
                                                                         "batch_size"]])

                # Optimise critic
                loss_critic_val, _ = self.sess.run(
                    [self.critic_network.critic_loss, self.critic_network.optimizer],
                    feed_dict={self.critic_network.observation: observation_batch,
                               self.critic_network.prev_actions: previous_action_batch,
                               self.critic_network.internal_state: internal_state_batch,
                               self.critic_network.rnn_state_in: critic_rnn_state_slice,
                               self.critic_network.rnn_state_in_ref: critic_rnn_state_ref_slice,

                               self.critic_network.returns_placeholder: return_batch,

                               self.critic_network.trainLength: self.learning_params["trace_length"],
                               self.critic_network.batch_size: current_batch_size,
                               self.critic_network.learning_rate: self.learning_params[
                                                                      "learning_rate_critic"] * current_batch_size

                               })

                # Optimise actor
                loss_actor_val_impulse, loss_actor_val_angle, _ = self.sess.run(
                    [self.actor_network.impulse_loss, self.actor_network.angle_loss,
                     self.actor_network.optimizer],
                    feed_dict={self.actor_network.observation: observation_batch,
                               self.actor_network.prev_actions: previous_action_batch,
                               self.actor_network.internal_state: internal_state_batch,
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,
                               self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined: self.angle_sigma,

                               self.actor_network.impulse_placeholder: impulse_batch,
                               self.actor_network.angle_placeholder: angle_batch,
                               self.actor_network.old_log_prob_impulse_placeholder: log_impulse_probability_batch,
                               self.actor_network.old_log_prob_angle_placeholder: log_angle_probability_batch,
                               self.actor_network.scaled_advantage_placeholder: advantage_batch,

                               self.actor_network.trainLength: self.learning_params["trace_length"],
                               self.actor_network.batch_size: current_batch_size,
                               self.actor_network.learning_rate: self.learning_params[
                                                                     "learning_rate_actor"] * current_batch_size
                               })

                average_loss_impulse += np.mean(np.abs(loss_actor_val_impulse))
                average_loss_angle += np.mean(np.abs(loss_actor_val_angle))
                average_loss_value += np.abs(loss_critic_val)

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])

    def train_network_multivariate(self):
        observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, \
        log_action_probability_buffer, advantage_buffer, return_buffer, \
        key_rnn_points = self.buffer.get_episode_buffer()

        number_of_batches = int(math.ceil(observation_buffer.shape[0] / self.learning_params["batch_size"]))

        for batch in range(number_of_batches):
            # Find steps at start of each trace to compute RNN states
            batch_key_points = [i for i in key_rnn_points if
                                batch * self.learning_params["batch_size"] * self.learning_params[
                                    "trace_length"] <= i < (batch + 1) *
                                self.learning_params["batch_size"] * self.learning_params["trace_length"]]

            # Get the current batch
            observation_batch, internal_state_batch, action_batch, previous_action_batch, \
            log_action_probability_batch, advantage_batch, \
            return_batch, current_batch_size = self.get_batch_multivariate(batch, observation_buffer,
                                                                           internal_state_buffer,
                                                                           action_buffer, previous_action_buffer,
                                                                           log_action_probability_buffer,
                                                                           advantage_buffer,
                                                                           return_buffer)

            # Loss value logging
            average_loss_value = 0
            average_loss_impulse = 0
            average_loss_angle = 0

            for i in range(self.learning_params["n_updates_per_iteration"]):
                # Compute RNN states for start of each trace.
                actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, \
                critic_rnn_state_ref_slice = self.compute_rnn_states(batch_key_points,
                                                                     observation_buffer[
                                                                     :(batch + 1) * self.learning_params["batch_size"]],
                                                                     internal_state_buffer[
                                                                     :(batch + 1) * self.learning_params["batch_size"]],
                                                                     previous_action_buffer[
                                                                     :(batch + 1) * self.learning_params[
                                                                         "batch_size"]])

                # Optimise critic
                loss_critic_val, _ = self.sess.run(
                    [self.critic_network.critic_loss, self.critic_network.optimizer],
                    feed_dict={self.critic_network.observation: observation_batch,
                               self.critic_network.prev_actions: previous_action_batch,
                               self.critic_network.internal_state: internal_state_batch,
                               self.critic_network.rnn_state_in: critic_rnn_state_slice,
                               self.critic_network.rnn_state_in_ref: critic_rnn_state_ref_slice,

                               self.critic_network.returns_placeholder: return_batch,

                               self.critic_network.trainLength: self.learning_params["trace_length"],
                               self.critic_network.batch_size: current_batch_size,
                               self.critic_network.learning_rate: self.learning_params[
                                                                      "learning_rate_critic"] * current_batch_size

                               })

                # Optimise actor
                loss_actor_val, _ = self.sess.run(
                    [self.actor_network.total_loss,
                     self.actor_network.optimizer],
                    feed_dict={self.actor_network.observation: observation_batch,
                               self.actor_network.prev_actions: previous_action_batch,
                               self.actor_network.internal_state: internal_state_batch,
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,

                               self.actor_network.action_placeholder: action_batch,
                               self.actor_network.old_log_prob: log_action_probability_batch,
                               self.actor_network.scaled_advantage_placeholder: advantage_batch,

                               self.actor_network.trainLength: self.learning_params["trace_length"],
                               self.actor_network.batch_size: current_batch_size,
                               self.actor_network.learning_rate: self.learning_params[
                                                                     "learning_rate_actor"] * current_batch_size
                               })

                average_loss_impulse += np.mean(np.abs(loss_actor_val))
                average_loss_angle += np.mean(np.abs(loss_actor_val))
                average_loss_value += np.abs(loss_critic_val)

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])

    def old_training(self):
        number_of_batches = 5
        for batch in range(number_of_batches):
            if batch == number_of_batches - 1:
                final_batch = True
                current_batch_size = len(self.buffer.return_buffer) - self.buffer.pointer

            else:
                final_batch = False
                current_batch_size = self.learning_params["batch_size"]
            observation_slice, internal_state_slice, action_slice, previous_action_slice, reward_slice, value_slice, \
            log_impulse_probability_slice, log_angle_probability_slice, advantage_slice, return_slice, \
            actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, critic_rnn_state_ref_slice = self.buffer.get_batch(
                final_batch)

            # Moveed to buffer
            # actor_rnn_state_slice = np.moveaxis(actor_rnn_state_slice, 1, 0).squeeze()[:, 0:1, :]
            # actor_rnn_state_ref_slice = np.moveaxis(actor_rnn_state_ref_slice, 1, 0).squeeze()[:, 0:1, :]
            # critic_rnn_state_slice = np.moveaxis(critic_rnn_state_slice, 1, 0).squeeze()[:, 0:1, :]
            # critic_rnn_state_ref_slice = np.moveaxis(critic_rnn_state_ref_slice, 1, 0).squeeze()[:, 0:1, :]
            #
            # actor_rnn_state_slice = (actor_rnn_state_slice[0], actor_rnn_state_slice[1])
            # actor_rnn_state_ref_slice = (actor_rnn_state_ref_slice[0], actor_rnn_state_ref_slice[1])
            # critic_rnn_state_slice = (critic_rnn_state_slice[0], critic_rnn_state_slice[1])
            # critic_rnn_state_ref_slice = (critic_rnn_state_ref_slice[0], critic_rnn_state_ref_slice[1])

            average_loss_value = 0
            average_loss_impulse = 0
            average_loss_angle = 0
            for i in range(self.learning_params["n_updates_per_iteration"]):
                # Dont Recompute initial state
                loss_critic_val, _ = self.sess.run(
                    [self.critic_network.critic_loss, self.critic_network.optimizer],
                    feed_dict={self.critic_network.observation: np.vstack(observation_slice),
                               # self.critic_network.scaler: np.full(np.vstack(observation_slice).shape, 255),
                               self.critic_network.prev_actions: np.vstack(previous_action_slice),
                               self.critic_network.internal_state: np.vstack(internal_state_slice),
                               self.critic_network.rnn_state_in: critic_rnn_state_slice,
                               self.critic_network.rnn_state_in_ref: critic_rnn_state_ref_slice,

                               self.critic_network.returns_placeholder: np.vstack(return_slice).flatten(),

                               self.critic_network.trainLength: current_batch_size,
                               self.critic_network.batch_size: 1,
                               })

                loss_actor_val_impulse, loss_actor_val_angle, _ = self.sess.run(
                    [self.actor_network.impulse_loss, self.actor_network.angle_loss,
                     self.actor_network.optimizer],
                    feed_dict={self.actor_network.observation: np.vstack(observation_slice),
                               # self.actor_network.scaler: np.full(np.vstack(observation_slice).shape, 255),
                               self.actor_network.prev_actions: np.vstack(previous_action_slice),
                               self.actor_network.internal_state: np.vstack(internal_state_slice),
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,
                               self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined: self.angle_sigma,

                               self.actor_network.impulse_placeholder: np.vstack(action_slice[:, 0]),
                               self.actor_network.angle_placeholder: np.vstack(action_slice[:, 1]),
                               self.actor_network.old_log_prob_impulse_placeholder: log_impulse_probability_slice.flatten(),
                               self.actor_network.old_log_prob_angle_placeholder: log_angle_probability_slice.flatten(),
                               self.actor_network.scaled_advantage_placeholder: np.vstack(advantage_slice).flatten(),

                               self.actor_network.trainLength: current_batch_size,
                               self.actor_network.batch_size: 1,
                               })

                average_loss_impulse += np.mean(np.abs(loss_actor_val_impulse))
                average_loss_angle += np.mean(np.abs(loss_actor_val_angle))
                average_loss_value += np.abs(loss_critic_val)

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])

